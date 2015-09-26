package MyJ48;

import weka.core.Attribute;
import weka.core.Instances;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class Splitable extends NodeType{
    /**
     * attribute that is going to be used for splitting
     */
    public Attribute attribute;

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
    public double ratioGain;

    /**
     * Number of branches created from this node
     */
    public double numberOfBranch;

    /**
     * Number of posible splits
     */
    public double numberOfSplitPoints;

    public Splitable(Attribute splitAttribute, double minimalInstances, double totalWeight)
    {
        this.attribute = splitAttribute;
        this.minimalInstances = minimalInstances;
        this.totalWeight = totalWeight;
    }

    public void buildClassifier(Instances dataset)
    {
        this.dataset = dataset;
        numOfSubsets = 0;
        splitPointValue = Double.MAX_VALUE;
        infoGain = 0;
        ratioGain = 0;
        numberOfSplitPoints = 0;

        /* different handling for nominal and numeric attributes */
        if(attribute.isNominal())
        {
            numberOfBranch = attribute.numValues();
            numberOfSplitPoints = attribute.numValues();
        }
        else // attribute == numeric
        {
            numberOfBranch = 2;
            numberOfSplitPoints = 0;
        }
    }

    private void handleNominalAttribute()
    {
        
    }

    private void handleNumericAttribute()
    {

    }
}
