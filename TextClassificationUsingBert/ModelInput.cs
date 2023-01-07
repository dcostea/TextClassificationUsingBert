using Microsoft.ML.Data;

public class ModelInput
{
    public ModelInput(string utterance)
    {
        Content = utterance;
    }

    [LoadColumn(0)]
    [ColumnName(@"Content")]
    public string Content { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Label")]
    public string Label { get; set; }

}
