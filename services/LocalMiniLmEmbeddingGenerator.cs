using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.HuggingFace.Tokenizer;

namespace VectorDataAI.services;

public sealed class LocalMiniLmEmbeddingGenerator :
IEmbeddingGenerator<string, Embedding<float>>,
IDisposable
{
    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;

public LocalMiniLmEmbeddingGenerator(
    string modelPath,
    string tokenizerPath)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"ONNX model not found: {modelPath}");
        }

        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException(
                $"Tokenizer not found: {tokenizerPath}");
        }

        Console.WriteLine("Loading ONNX model...");

        _session = new InferenceSession(modelPath);

        Console.WriteLine("Loading tokenizer...");

        _tokenizer = Tokenizer.FromFile(tokenizerPath);

        Console.WriteLine();
        Console.WriteLine("ONNX inputs:");

        foreach (var input in _session.InputMetadata)
        {
            Console.WriteLine(
                $"  {input.Key} | " +
                $"Type: {input.Value.ElementType} | " +
                $"Shape: [{string.Join(", ", input.Value.Dimensions)}]");
        }

        Console.WriteLine();
        Console.WriteLine("ONNX outputs:");

        foreach (var output in _session.OutputMetadata)
        {
            Console.WriteLine(
                $"  {output.Key} | " +
                $"Type: {output.Value.ElementType} | " +
                $"Shape: [{string.Join(", ", output.Value.Dimensions)}]");
        }

        Console.WriteLine();
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<Embedding<float>>();

        foreach (string value in values)
        {
            cancellationToken.ThrowIfCancellationRequested();

            float[] vector = GenerateEmbedding(value);

            results.Add(new Embedding<float>(vector));
        }

        return Task.FromResult(
            new GeneratedEmbeddings<Embedding<float>>(results));
    }

    private float[] GenerateEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException(
                "Text cannot be empty.",
                nameof(text));
        }

        // ---------------------------------------------------------
        // Tokenization
        // ---------------------------------------------------------

        // The tokenizer API returns a collection of encodings for the input.
        // Take the first encoding for single-string input.
        var encodings = _tokenizer.Encode(text, addSpecialTokens: true);
        var encoding = encodings.First();

        var ids = encoding.Ids
            .Select(x => (long)x)
            .ToArray();

        const int maxLength = 256;

        if (ids.Length > maxLength)
        {
            ids = ids.Take(maxLength).ToArray();
        }

        if (ids.Length == 0)
        {
            throw new InvalidOperationException(
                "Tokenizer produced zero tokens.");
        }

        // ---------------------------------------------------------
        // Attention mask
        // ---------------------------------------------------------

        long[] attentionMask;

        if (encoding.AttentionMask is not null)
        {
            attentionMask = encoding.AttentionMask
                .Take(ids.Length)
                .Select(x => (long)x)
                .ToArray();
        }
        else
        {
            attentionMask = Enumerable
                .Repeat(1L, ids.Length)
                .ToArray();
        }

        // Make sure both arrays have identical length.
        if (attentionMask.Length != ids.Length)
        {
            attentionMask = Enumerable
                .Repeat(1L, ids.Length)
                .ToArray();
        }

        // ---------------------------------------------------------
        // Token type IDs
        // ---------------------------------------------------------

        long[] tokenTypeIds =
            Enumerable
                .Repeat(0L, ids.Length)
                .ToArray();

        var shape = new[] { 1, ids.Length };

        // Create tensors with the given shape and copy values into them.
        var inputIdsTensor = new DenseTensor<long>(shape);
        for (int i = 0; i < ids.Length; i++)
        {
            inputIdsTensor[0, i] = ids[i];
        }

        var attentionMaskTensor = new DenseTensor<long>(shape);
        for (int i = 0; i < attentionMask.Length; i++)
        {
            attentionMaskTensor[0, i] = attentionMask[i];
        }

        var tokenTypeIdsTensor = new DenseTensor<long>(shape);
        for (int i = 0; i < tokenTypeIds.Length; i++)
        {
            tokenTypeIdsTensor[0, i] = tokenTypeIds[i];
        }

        // ---------------------------------------------------------
        // Create ONNX inputs
        // ---------------------------------------------------------

        var inputs = new List<NamedOnnxValue>();

        if (_session.InputMetadata.ContainsKey("input_ids"))
        {
            inputs.Add(
                NamedOnnxValue.CreateFromTensor(
                    "input_ids",
                    inputIdsTensor));
        }

        if (_session.InputMetadata.ContainsKey("attention_mask"))
        {
            inputs.Add(
                NamedOnnxValue.CreateFromTensor(
                    "attention_mask",
                    attentionMaskTensor));
        }

        if (_session.InputMetadata.ContainsKey("token_type_ids"))
        {
            inputs.Add(
                NamedOnnxValue.CreateFromTensor(
                    "token_type_ids",
                    tokenTypeIdsTensor));
        }

        // ---------------------------------------------------------
        // Run ONNX model
        // ---------------------------------------------------------

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
            _session.Run(inputs);

        // ---------------------------------------------------------
        // Print outputs on first execution
        // ---------------------------------------------------------

        Console.WriteLine(
            $"Generated embedding for: \"{text}\"");

        foreach (var output in outputs)
        {
            Console.WriteLine(
                $"Output: {output.Name}");
        }

        // ---------------------------------------------------------
        // Look for sentence embedding
        // ---------------------------------------------------------

        var sentenceEmbedding =
            outputs.FirstOrDefault(x =>
                x.Name.Contains(
                    "sentence_embedding",
                    StringComparison.OrdinalIgnoreCase));

        if (sentenceEmbedding is not null)
        {
            return sentenceEmbedding
                .AsTensor<float>()
                .ToArray();
        }

        // ---------------------------------------------------------
        // Look for embedding output
        // ---------------------------------------------------------

        var embeddingOutput =
            outputs.FirstOrDefault(x =>
                x.Name.Equals(
                    "embedding",
                    StringComparison.OrdinalIgnoreCase));

        if (embeddingOutput is not null)
        {
            return embeddingOutput
                .AsTensor<float>()
                .ToArray();
        }

        // ---------------------------------------------------------
        // Look for last hidden state
        // ---------------------------------------------------------

        var hiddenOutput =
            outputs.FirstOrDefault(x =>
                x.Name.Contains(
                    "last_hidden_state",
                    StringComparison.OrdinalIgnoreCase));

        if (hiddenOutput is null)
        {
            // If we cannot identify the output, use the first
            // float tensor as a fallback.
            hiddenOutput =
                outputs.FirstOrDefault(x =>
                    x.AsTensor<float>() is not null);
        }

        if (hiddenOutput is null)
        {
            throw new InvalidOperationException(
                "Could not find an embedding output in the ONNX model.");
        }

        var hidden =
            hiddenOutput.AsTensor<float>();

        // ---------------------------------------------------------
        // If already [1,384], return directly
        // ---------------------------------------------------------

        if (hidden.Dimensions.Length == 2 &&
            hidden.Dimensions[0] == 1 &&
            hidden.Dimensions[1] == 384)
        {
            return hidden.ToArray();
        }

        // ---------------------------------------------------------
        // Mean pooling
        // ---------------------------------------------------------

        if (hidden.Dimensions.Length != 3)
        {
            throw new InvalidOperationException(
                $"Unexpected ONNX output shape: " +
                $"[{string.Join(", ", hidden.Dimensions.ToArray())}]");
        }

        int sequenceLength =
            hidden.Dimensions[1];

        int dimensions =
            hidden.Dimensions[2];

        float[] embedding =
            new float[dimensions];

        float tokenCount = 0;

        for (int token = 0;
             token < sequenceLength;
             token++)
        {
            if (token >= attentionMask.Length)
                break;

            if (attentionMask[token] == 0)
                continue;

            tokenCount++;

            for (int dimension = 0;
                 dimension < dimensions;
                 dimension++)
            {
                embedding[dimension] +=
                    hidden[0, token, dimension];
            }
        }

        if (tokenCount == 0)
        {
            throw new InvalidOperationException(
                "Attention mask contains no active tokens.");
        }

        for (int dimension = 0;
             dimension < dimensions;
             dimension++)
        {
            embedding[dimension] /=
                tokenCount;
        }

        // ---------------------------------------------------------
        // L2 normalization
        // ---------------------------------------------------------

        float magnitude =
            MathF.Sqrt(
                embedding.Sum(x => x * x));

        if (magnitude > 0)
        {
            for (int i = 0;
                 i < embedding.Length;
                 i++)
            {
                embedding[i] /=
                    magnitude;
            }
        }

        return embedding;
    }

    public object? GetService(
        Type serviceType,
        object? serviceKey)
    {
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(EmbeddingGeneratorMetadata))
        {
            // EmbeddingGeneratorMetadata constructor may not include an 'endpoint' named parameter
            // Use the common (providerName, modelId) constructor overload.
            return new EmbeddingGeneratorMetadata(
                "LocalMiniLmEmbeddingGenerator",
                null);
        }

        return serviceType.IsInstanceOfType(this)
            ? this
            : null;
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
        _session.Dispose();
    }


}
