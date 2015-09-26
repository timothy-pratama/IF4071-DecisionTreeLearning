package MyJ48;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class NotSplitable extends NodeType{

    public NotSplitable(J48ClassDistribution distribution)
    {
        distribution = new J48ClassDistribution(distribution);
        numOfSubsets = 1;
    }
}
