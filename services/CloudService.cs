using Microsoft.Extensions.VectorData;
using System;
using System.Collections.Generic;
using System.Text;

namespace VectorDataAI.services
{

    public class CloudService
    {
        [VectorStoreKey]
        public int Key { get; set; }

        [VectorStoreData]
        public string Name { get; set; } = string.Empty;

        [VectorStoreData]
        public string Description { get; set; } = string.Empty;

        [VectorStoreVector(
        dimensions: 1536,
        DistanceFunction = DistanceFunction.CosineSimilarity)]
        public ReadOnlyMemory<float> Vector { get; set; }
    }
}
