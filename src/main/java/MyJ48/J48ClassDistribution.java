package MyJ48;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class J48ClassDistribution {

    /* Weight of instances per subdataset per class. */
    public double weightPerClassPerSubdataset[][];

    /* Weight of instances per subdataset */
    public double weightPerSubDataset[];

    /* Weight of instances per class. */
    public double weightPerClass[];

    /* Total weight of instances. */
    public double weightTotal;

    /**
     * Create distributions with one dataset (since it's the first time).
     * @param dataSet
     */
    public J48ClassDistribution(Instances dataSet)
    {
        weightTotal = 0;
        weightPerClassPerSubdataset = new double[1][dataSet.numClasses()];
        weightPerSubDataset = new double[1];
        weightPerClass = new double[dataSet.numClasses()];

        Enumeration instancesEnumeration = dataSet.enumerateInstances();
        while(instancesEnumeration.hasMoreElements())
        {
            Instance i = (Instance) instancesEnumeration.nextElement();
            addInstanceToDataset(0, i);
        }
    }

    /**
     * Create a single subdataset from target distribution
     * @param targetDistribution
     * @return
     */
    public J48ClassDistribution(J48ClassDistribution targetDistribution)
    {
        weightTotal = targetDistribution.weightTotal;
        weightPerClassPerSubdataset = new double[1][targetDistribution.numClasses()];
        weightPerSubDataset = new double[1];
        weightPerClass = new double[targetDistribution.numClasses()];

        for(int i = 0; i < targetDistribution.numClasses(); i++)
        {
            weightPerClassPerSubdataset[0][i] = targetDistribution.weightPerClass[i];
            weightPerClass[i] = targetDistribution.weightPerClass[i];
        }
        weightPerSubDataset[0] = targetDistribution.weightTotal;
    }

    /**
     * return the number of subdataset in this distribution
     * @return
     */
    private int numSubDatasets()
    {
        return weightPerSubDataset.length;
    }

    /**
     * return the number of class in this distribution
     * @return
     */
    private int numClasses()
    {
        return weightPerClass.length;
    }

    /**
     * add single instance to subdataset
     * @param subDatasetIndex
     * @param instance
     */
    private void addInstanceToDataset(int subDatasetIndex, Instance instance)
    {
        int classIndex = (int) instance.classValue();
        weightPerClassPerSubdataset[subDatasetIndex][classIndex] = weightPerClassPerSubdataset[subDatasetIndex][classIndex] + instance.weight();
        weightPerSubDataset[subDatasetIndex] = weightPerSubDataset[subDatasetIndex] + instance.weight();
        weightPerClass[classIndex] = weightPerClass[classIndex] +  instance.weight();
        weightTotal = weightTotal + instance.weight();
    }
}
