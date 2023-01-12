using Microsoft.ML.Data;

namespace TextClassificationUsingBert.Models;

public class ModelOutput
{
    [ColumnName(@"PredictedLabel")]
    public string? PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[]? Score { get; set; }
}