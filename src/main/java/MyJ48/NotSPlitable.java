package MyJ48;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class NotSplitable extends NodeType{

    public NotSplitable(J48ClassDistribution distribution)
    {
        distribution = new J48ClassDistribution(distribution);
        numOfSubsets = 1;
    }

    @Override
    public int getSubsetIndex(Instance instance) {
        return 0;
    }

    @Override
    public double[] getWeights(Instance instance) {
        return null;
    }
}
