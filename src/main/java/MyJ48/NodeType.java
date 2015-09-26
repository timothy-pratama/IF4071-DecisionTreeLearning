package MyJ48;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class NodeType {
    public J48ClassDistribution classDistribution;
    public int numOfSubsets;

    public boolean validateNode()
    {
        return (numOfSubsets > 0);
    }
}
