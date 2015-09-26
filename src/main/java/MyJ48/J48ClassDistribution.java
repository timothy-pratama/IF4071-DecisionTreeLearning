package MyJ48;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class J48ClassDistribution {

    /* Weight of instances per subdataset per class. */
    public double weightClassPerSubdataset[][];

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
        weightClassPerSubdataset = new double[1][dataSet.numClasses()];
        weightPerSubDataset = new double[1];
        weightPerClass = new double[dataSet.numClasses()];

        Enumeration instancesEnumeration = dataSet.enumerateInstances();
        while(instancesEnumeration.hasMoreElements())
        {
            Instance i = (Instance) instancesEnumeration.nextElement();
            addInstance(0, i);
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
        weightClassPerSubdataset = new double[1][targetDistribution.numClasses()];
        weightPerSubDataset = new double[1];
        weightPerClass = new double[targetDistribution.numClasses()];

        for(int i = 0; i < targetDistribution.numClasses(); i++)
        {
            weightClassPerSubdataset[0][i] = targetDistribution.weightPerClass[i];
            weightPerClass[i] = targetDistribution.weightPerClass[i];
        }
        weightPerSubDataset[0] = targetDistribution.weightTotal;
    }

    public J48ClassDistribution(double numberOfBranch, int numberOfClass) {
        weightTotal = 0;
        weightClassPerSubdataset = new double[(int) numberOfBranch][numberOfClass];
        weightPerSubDataset = new double[(int) numberOfBranch];
        weightPerClass = new double[numberOfClass];
    }

    /**
     * return the number of subdataset in this distribution
     * @return
     */
    public int numSubDatasets()
    {
        return weightPerSubDataset.length;
    }

    /**
     * Return the total weight of the class distribution
     * @return
     */
    public double getTotalWeight()
    {
        return weightTotal;
    }

    /**
     * return the number of class in this distribution
     * @return
     */
    public int numClasses()
    {
        return weightPerClass.length;
    }

    /**
     * add single instance to subdataset
     * @param subDatasetIndex
     * @param instance
     */
    public void addInstance(int subDatasetIndex, Instance instance)
    {
        int classIndex = (int) instance.classValue();
        weightClassPerSubdataset[subDatasetIndex][classIndex] = weightClassPerSubdataset[subDatasetIndex][classIndex] + instance.weight();
        weightPerSubDataset[subDatasetIndex] = weightPerSubDataset[subDatasetIndex] + instance.weight();
        weightPerClass[classIndex] = weightPerClass[classIndex] +  instance.weight();
        weightTotal = weightTotal + instance.weight();
    }

    public boolean isSplitable(double minimalInstances)
    {
        int counter = 0;
        for(int i=0; i<weightPerSubDataset.length; i++)
        {
            if(Utils.grOrEq(weightPerSubDataset[i], minimalInstances))
            {
                counter ++;
            }
        }
        return (counter > 1);
    }

    public double computeInitialEntropy()
    {
        double initEntropy = 0;
        for(int i=0; i<numClasses(); i++)
        {
            double p = weightPerClass[i]/weightTotal;
            initEntropy = initEntropy + (p * log2(p));
        }
        initEntropy = initEntropy * -1;
        return initEntropy;
    }

    public double calculateInfoGain(double instancesTotalWeight)
    {
        /* initial entropy */
        /* entropy = -(p1 * log2 p1 + p2 * log2 p2 + ...) */
        double initialEntropy = 0;
        double unknownValues = 0;
        double unknownRate = 0;

        initialEntropy = computeInitialEntropy();
//        System.out.printf("=====Initial entropy: %f\n", initialEntropy);

        for (int i=0; i<numSubDatasets(); i++)
        {
            double finalEntropy = 0;
            for(int j=0; j<numClasses(); j++)
            {
                double p = 0;
                if(weightPerSubDataset[i] > 0)
                {
                    p = weightClassPerSubdataset[i][j] / weightPerSubDataset[i];
                }
                finalEntropy = finalEntropy + (p * log2(p));
            }
            finalEntropy = finalEntropy * -1;
            initialEntropy = initialEntropy - (weightPerSubDataset[i]/weightTotal*finalEntropy);
        }
//        System.out.println("=====Information Gain: " + initialEntropy);

        unknownValues = instancesTotalWeight-weightTotal;
        unknownRate = unknownRate/instancesTotalWeight;

//        System.out.println("=====Unknown Values: " + unknownValues);
//        System.out.println("=====Unknown Rate: " + unknownRate);
//        System.out.println("=====Information Gain Final: " + (1-unknownRate)*initialEntropy);

        return ((1-unknownRate)*initialEntropy);
    }

    public double calculateGainRatio(double infoGain) {

        /* splitInformation = -(p1 * log2 p1 + p2 * log2 p2 + ...) */
        double splitInformation = 0;
        for(int i=0; i<numSubDatasets(); i++)
        {
            double p = weightPerSubDataset[i]/weightTotal;
            splitInformation = splitInformation + (p * log2(p));
        }
        splitInformation = splitInformation * -1;
        if(Utils.eq(splitInformation,0))
        {
            return 0;
        }
        else
        {
            return infoGain / splitInformation;
        }
    }

    private double log2(double a) {
        if(a != 0)
        {
            return Math.log(a) / Math.log(2);
        }
        else
        {
            return 0;
        }
    }

    public void print() {
        System.out.print("=====Weight per subdataset: ");
        for(double d : weightPerSubDataset)
        {
            System.out.print(d + " ");
        }
        System.out.println();

        System.out.print("=====Weight per class: ");
        for(double d : weightPerClass)
        {
            System.out.print(d + " ");
        }
        System.out.println();

        System.out.println("===== WeightClassPerSubdataset:");
        for(int i=0; i<numSubDatasets(); i++)
        {
            System.out.printf("Dataset[%d]: %f %f\n",i,weightClassPerSubdataset[i][0], weightClassPerSubdataset[i][1]);
        }
    }
}
