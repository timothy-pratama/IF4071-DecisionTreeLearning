package MyJ48;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

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

    public int getSubsetIndex(Instance instance)
    {
        return -1;
    }

    public double [] getWeights(Instance instance)
    {
        return null;
    }

    public Instances[] split(Instances dataSet)
    {
        Instances [] subDataset = new Instances[numOfSubsets];
        double weights[];
        double newWeight;
        Instance instance;
        int subset;

        for(int i=0; i<numOfSubsets; i++)
        {
            subDataset[i] = new Instances(dataSet, dataSet.numInstances());
        }

        for(int i=0; i<dataSet.numInstances(); i++)
        {
            instance = dataSet.instance(i);
            weights = getWeights(instance);
            subset = getSubsetIndex(instance);
            if(subset > -1)
            {
                subDataset[subset].add(instance);
            }
            else
            {
                for(int j=0; j<numOfSubsets; j++)
                {
                    if(Utils.gr(weights[j],0))
                    {
                        newWeight = weights[j] * instance.weight();
                        subDataset[j].add(instance);
                        subDataset[j].lastInstance().setWeight(newWeight);
                    }
                }
            }
        }

        for(int j=0; j<numOfSubsets; j++)
        {
            subDataset[j].compactify();
        }

        return subDataset;
    }
}
