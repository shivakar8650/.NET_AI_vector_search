using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using VectorDataAI;
using VectorDataAI.services;

// ---------------------------------------------------------
// 1. Sample Azure services
// ---------------------------------------------------------

List<CloudService> cloudServices =
[
new()
{
Key = 0,
Name = "Azure App Service",
Description =
"Host .NET, Java, Node.js, and Python web applications and APIs " +
"in a fully managed Azure service. You only need to deploy your " +
"code to Azure. Azure takes care of infrastructure management " +
"like high availability, load balancing, and autoscaling."
},

new()
{
    Key = 1,
    Name = "Azure Service Bus",
    Description =
        "A fully managed enterprise message broker supporting both " +
        "point-to-point and publish-subscribe integrations. It is ideal " +
        "for building decoupled applications, queue-based load leveling, " +
        "or facilitating communication between microservices."
},

new()
{
    Key = 2,
    Name = "Azure Blob Storage",
    Description =
        "Azure Blob Storage allows your applications to store and retrieve " +
        "files in the cloud. Azure Storage is highly scalable for storing " +
        "massive amounts of data and data is stored redundantly to ensure " +
        "high availability."
},

new()
{
    Key = 3,
    Name = "Microsoft Entra ID",
    Description =
        "Manage user identities and control access to your apps, data, " +
        "and resources."
},

new()
{
    Key = 4,
    Name = "Azure Key Vault",
    Description =
        "Store and access application secrets like connection strings " +
        "and API keys in an encrypted vault with restricted access to " +
        "make sure your secrets and your application are not compromised."
},

new()
{
    Key = 5,
    Name = "Azure AI Search",
    Description =
        "Information retrieval at scale for traditional and conversational " +
        "search applications, with security and options for AI enrichment " +
        "and vectorization."
}

];

// ---------------------------------------------------------
// 2. Create local ONNX embedding generator
// ---------------------------------------------------------

string modelPath = Path.Combine(
AppContext.BaseDirectory,
"Models",
"all-MiniLM-L6-v2-onnx",
"model.onnx");

string tokenizerPath = Path.Combine(
AppContext.BaseDirectory,
"Models",
"all-MiniLM-L6-v2-onnx",
"tokenizer.json");

Console.WriteLine("Checking local model...");

if (!File.Exists(modelPath))
{
    Console.WriteLine($"ONNX model not found:");
    Console.WriteLine(modelPath);
    return;
}

if (!File.Exists(tokenizerPath))
{
    Console.WriteLine($"Tokenizer not found:");
    Console.WriteLine(tokenizerPath);
    return;
}

Console.WriteLine($"Model: {modelPath}");
Console.WriteLine($"Tokenizer: {tokenizerPath}");
Console.WriteLine();

using IEmbeddingGenerator<string, Embedding<float>> generator =
new LocalMiniLmEmbeddingGenerator(
modelPath,
tokenizerPath);

// ---------------------------------------------------------
// 3. Create in-memory vector store
// ---------------------------------------------------------

var vectorStore = new InMemoryVectorStore();

// ---------------------------------------------------------
// 4. Create collection
// ---------------------------------------------------------

VectorStoreCollection<int, CloudService> cloudServicesStore =
vectorStore.GetCollection<int, CloudService>(
"cloudServices");

// ---------------------------------------------------------
// 5. Create collection if it doesn't exist
// ---------------------------------------------------------

await cloudServicesStore.EnsureCollectionExistsAsync();

// ---------------------------------------------------------
// 6. Generate embeddings and store records
// ---------------------------------------------------------

Console.WriteLine("Creating embeddings...");
Console.WriteLine();

foreach (CloudService service in cloudServices)
{
    Console.WriteLine($"Embedding: {service.Name}");

service.Vector =
    await generator.GenerateVectorAsync(
        service.Description);

    Console.WriteLine(
        $"Vector dimensions: {service.Vector.Length}");

    await cloudServicesStore.UpsertAsync(service);

}

Console.WriteLine();
Console.WriteLine("All services stored.");
Console.WriteLine();

// ---------------------------------------------------------
// 7. User search query
// ---------------------------------------------------------

string query =
"Which Azure service should I use to store my Word documents?";

Console.WriteLine($"Query: {query}");
Console.WriteLine();

// ---------------------------------------------------------
// 8. Convert query into embedding
// ---------------------------------------------------------

ReadOnlyMemory<float> queryEmbedding =
await generator.GenerateVectorAsync(query);

Console.WriteLine(
$"Query vector dimensions: {queryEmbedding.Length}");

Console.WriteLine();

// ---------------------------------------------------------
// 9. Search vector store
// ---------------------------------------------------------

IAsyncEnumerable<VectorSearchResult<CloudService>> results =
cloudServicesStore.SearchAsync(
queryEmbedding,
top: 3);

// ---------------------------------------------------------
// 10. Display results
// ---------------------------------------------------------

Console.WriteLine("Search results:");
Console.WriteLine("----------------");

await foreach (
VectorSearchResult<CloudService> result
in results)
{
    Console.WriteLine($"Name: {result.Record.Name}");
    Console.WriteLine($"Description: {result.Record.Description}");
    Console.WriteLine($"Vector match score: {result.Score}");
    Console.WriteLine();
}

Console.WriteLine("Done.");
