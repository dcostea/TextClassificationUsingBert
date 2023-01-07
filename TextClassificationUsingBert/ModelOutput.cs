using Microsoft.ML.Data;

public class ModelOutput
{
    [ColumnName(@"PredictedLabel")]
    public string PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[] Score { get; set; }
}

public class ModelIntermediary
{
    [ColumnName(@"Content")]
    public string Content { get; set; }

    [ColumnName(@"WordsWithoutDefaultStopWords")]
    public string[] WordsWithoutDefaultStopWords { get; set; }
}
