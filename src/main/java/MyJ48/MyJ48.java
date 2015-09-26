package MyJ48;

import Util.Util;
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

    }

    private NodeType processNode()
    {
        double minResult;
        Splitable[] splitables;
        Splitable bestSplitable = null;
        NotSplitable notSplitable = null;
        double averageInfoGain = 0;
        int usefulSplitables = 0;
        J48ClassDistribution distribution;
        double totalWeight;

        try{
            distribution = new J48ClassDistribution(dataSet);
            notSplitable = new NotSplitable(distribution);

            /* if there are not enough instances for splitting */
            /* if the data set only belong to 1 class */
            /* Then can't split this node much further */
            if(Utils.sm(dataSet.numInstances(), 2 * minimalInstances) ||
               Utils.eq(distribution.weightTotal, distribution.weightPerClass[Utils.maxIndex(distribution.weightPerClass)]))
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
            }
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
