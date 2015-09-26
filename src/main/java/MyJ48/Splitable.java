package MyJ48;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class Splitable extends NodeType{
    /**
     * attribute that is going to be used for splitting
     */
    public Attribute splitAttribute;

    /**
     * minimal instances required for splitting an attribute
     */
    public double minimalInstances;

    /**
     * total weight (in this case instances) for this splitable
     */
    public double totalWeight;

    /**
     * The dataset that are going to be used to train this node
     */
    Instances dataset;

    /**
     * Split the dataset according to this value
     */
    public double splitPointValue;

    /**
     * This node information gain
     */
    public double infoGain;

    /**
     * This node gain ratio
     */
    public double gainRatio;

    /**
     * Number of branches created from this node
     */
    public int numberOfBranch;

    /**
     * Number of posible splits
     */
    public int numberOfSplitPoints;

    /**
     * This node class distribution
     */
    public J48ClassDistribution classDistribution;

    public Splitable(Attribute splitAttribute, double minimalInstances, double totalWeight)
    {
        this.splitAttribute = splitAttribute;
        this.minimalInstances = minimalInstances;
        this.totalWeight = totalWeight;
    }

    public void buildClassifier(Instances dataset)
    {
//        System.out.println("=====Datasets: \n" + dataset);
        System.out.println("\n=====Current Attributes: " + splitAttribute.toString());
        this.dataset = dataset;
        numOfSubsets = 0;
        splitPointValue = Double.MAX_VALUE;
        infoGain = 0;
        gainRatio = 0;
        numberOfSplitPoints = 0;

        if(splitAttribute.isNominal())
        {
            System.out.println("=====Nominal Attribute!");
            numberOfBranch = splitAttribute.numValues();
            numberOfSplitPoints = splitAttribute.numValues();
            processNominalAttribute();
        }
        else // attribute == numeric
        {
            System.out.println("=====Numeric Attributes!");
            numberOfBranch = 2;
            numberOfSplitPoints = 0;
            dataset.sort(splitAttribute);
            processNumericAttribute();
        }
    }

    private void processNominalAttribute()
    {
        classDistribution = new J48ClassDistribution(numberOfBranch, dataset.numClasses());
        Enumeration instanceEnumeration = dataset.enumerateInstances();
        while(instanceEnumeration.hasMoreElements())
        {
            Instance instance = (Instance) instanceEnumeration.nextElement();
            if(!instance.isMissing(splitAttribute))
            {
                classDistribution.addInstanceToDataset((int) instance.value(splitAttribute), instance);
            }
        }

        if(classDistribution.isSplitable(minimalInstances))
        {
            System.out.println("=====Splitable!");
            numOfSubsets = numberOfBranch;
            infoGain = classDistribution.calculateInfoGain(totalWeight);
            System.out.println("=====Information Gain: " + infoGain);
            gainRatio = classDistribution.calculateGainRatio(infoGain);
            System.out.println("=====Gain Ratio: " + gainRatio);
        }
    }

    private void processNumericAttribute()
    {

    }
}
