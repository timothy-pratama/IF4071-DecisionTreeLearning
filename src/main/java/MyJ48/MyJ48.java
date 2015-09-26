package MyJ48;

import Util.Util;
import org.w3c.dom.Attr;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Enumeration;

/**
 * Created by timothy.pratama on 24-Sep-15.
 */
public class MyJ48 extends Classifier {

    /**
     * The children of this node
     */
    private MyJ48 [] childs;

    /**
     * this attribute store whether this node is leaf or not
     */
    private boolean is_leaf;

    /**
     * this attribute store whether this node is empty or not
     */
    private boolean is_empty;

    /**
     * this attribute store the data set used for training this model
     */
    private Instances dataSet;

    /**
     * This attribute is the minimal number of instances allowed for C4.5
     */
    private double minimalInstances = 2;

    /**
     *
     */
    private J48ClassDistribution testSetDistribution;

    /**
     * this attribute store the confidence level of the j48 tree
     */
    private float confidenceLevel = 0.25f;

    /**
     * This attribute store the type of this node whether it's splitable or not-splitable (leaf)
     */
    private NodeType nodeType;

    /**
     * Subdataset for this node
     */
    Instances [] subDataset;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Check if the data set is able to be proccessed using MyJ48.MyJ48
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();

        createTree(data);
    }

    private void createTree(Instances data)
    {
        dataSet = data;
        is_leaf = false;
        is_empty = false;
        testSetDistribution = null;

        nodeType = processNode();
        if(nodeType.numOfSubsets > 1)
        {
            subDataset = nodeType.split(dataSet);
//            dataSet = null;
            childs = new MyJ48[nodeType.numOfSubsets];
            for(int i=0; i<nodeType.numOfSubsets; i++)
            {
                childs[i] = createNewTree(subDataset[i]);
            }
        }
        else
        {
            is_leaf = true;
            if(Utils.eq(dataSet.sumOfWeights(), 0))
            {
                is_empty = true;
            }
        }
    }

    private MyJ48 createNewTree(Instances subDataset) {
        MyJ48 newMyJ48 = new MyJ48();
        newMyJ48.createTree(subDataset);
        return newMyJ48;
    }

    private NodeType processNode()
    {
        double minGainRatio;
        Splitable[] splitables;
        Splitable bestSplitable = null;
        NotSplitable notSplitable = null;
        double averageInfoGain = 0;
        int usefulSplitables = 0;
        J48ClassDistribution classDistribution;
        double totalWeight;

        try{
            classDistribution = new J48ClassDistribution(dataSet);
            notSplitable = new NotSplitable(classDistribution);

            /* if there are not enough instances for splitting */
            /* if the data set only belong to 1 class */
            /* Then can't split this node much further */
            if(Utils.sm(dataSet.numInstances(), 2 * minimalInstances) ||
               Utils.eq(classDistribution.weightTotal, classDistribution.weightPerClass[Utils.maxIndex(classDistribution.weightPerClass)]))
            {
                return notSplitable;
            }

            /* The node is splitable */

            splitables = new Splitable[dataSet.numAttributes()];
            totalWeight = dataSet.sumOfWeights();

            Enumeration attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                splitables[attribute.index()] = new Splitable(attribute, minimalInstances, dataSet.sumOfWeights());
                splitables[attribute.index()].buildClassifier(dataSet);
                if(splitables[attribute.index()].validateNode())
                {
                    if(dataSet != null)
                    {
                        averageInfoGain = averageInfoGain +  splitables[attribute.index()].infoGain;
                        usefulSplitables++;
                    }
                }
            }

            if (usefulSplitables == 0)
            {
                return notSplitable;
            }
            averageInfoGain = averageInfoGain/(double)usefulSplitables;

            minGainRatio = 0;
            attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                if(splitables[attribute.index()].validateNode())
                {
                    if(splitables[attribute.index()].infoGain >= (averageInfoGain - 0.001) &&
                       Utils.gr(splitables[attribute.index()].gainRatio, minGainRatio))
                    {
                        bestSplitable = splitables[attribute.index()];
                        minGainRatio = bestSplitable.gainRatio;
                    }
                }
            }

            if (Utils.eq(minGainRatio,0))
            {
                return notSplitable;
            }

            bestSplitable.addInstanceWithMissingvalue();

            if(dataSet != null)
            {
                bestSplitable.setSplitPoint();
            }

            return bestSplitable;
        }

        catch (Exception e) {
            e.printStackTrace();
        };

        return null;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return super.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        /* Allowed attributes in MyJ48.MyJ48 */
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // Allowed class in MyJ48.MyJ48
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // Minimal instances for MyJ48.MyJ48
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public String toString() {
        return super.toString();
    }

    public static void main (String [] args) throws Exception {
        Classifier classifier = new MyJ48();
        Instances dataSet = Util.readARFF("weather.numeric.arff");
        classifier.buildClassifier(dataSet);
    }
}
