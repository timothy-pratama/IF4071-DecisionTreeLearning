import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Enumeration;

/**
 * Created by timothy.pratama on 21-Sep-15.
 */
public class MyId3 extends Classifier {

    /**
     * Attribut for serialization
     */
    static final long serialVersionUID = -2693273657194322561L;

    /**
     * Childs from an MyId3 Node
     */
    private MyId3[] childs;

    /**
     * Attribute for splitting. NULL if leaf, otherwise will be the best attribute for splitting
     */
    private Attribute splitAttribute;

    /**
     * The class index for the current node if it is a leaf.
     */
    private double classValue;

    /**
     * Attribute for storing a leaf's class distribution for determining class value.
     */
    private double[] ClassDistribution;

    /**
     * Attribute for the instances' Class possible values
     */
    private Attribute classAttribute;

    /**
     * Attribute for storing the most common value in the given data set
     */
    private double mostCommonClassValue;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        /* Make sure that there are no continues value are entered! */
        getCapabilities().testWithFail(instances);

        /* Delete all instances with missing class */
        /* not useful for building tree */
        Instances data = new Instances(instances);
        data.deleteWithMissingClass();

        /* Find the most commmon value for the given data set */
        double mostCommonClassValue;
        double [] classDistribution = new double[data.numClasses()];
        Enumeration dataEnum = data.enumerateInstances();
        while(dataEnum.hasMoreElements())
        {
            Instance instance = (Instance) dataEnum.nextElement();
            classDistribution[((int) instance.classValue())] ++;
        }
        mostCommonClassValue = Utils.maxIndex(classDistribution);

        createTree(data, mostCommonClassValue);
    }

    private void createTree(Instances dataSet, double mostCommonClassValue)
    {
        /* Initialization for several attributes */
        ClassDistribution = new double[dataSet.numClasses()];

        /* Check if there is no instances for this node -> possibly missing examples case */
        if(dataSet.numInstances() == 0)
        {
            splitAttribute = null;
            classValue = mostCommonClassValue;
        }
        else /* dataSet.numInstances() > 0 */
        {
            /* Compute each attribute information gain */
            double infoGains [] = new double[dataSet.numAttributes()];
            Enumeration attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                log("Attribute", attribute.toString());
                double infoGain = computeInfoGain(dataSet, attribute);
                infoGains[attribute.index()] = infoGain;
                System.out.println("==========Info Gain: " + infoGain);
            }
        }
    }

    private double computeInfoGain(Instances dataSet, Attribute attribute)
    {
        /* Info gain: Initial entropy - final entropy */
        /* Compute initial entropy */

        double initialEntropi = computeEntropy(dataSet);
        System.out.println("==========initial entropy: " + initialEntropi);

        Instances [] subDataSet = splitDataByAttribute(dataSet, attribute);
        System.out.println("==========Sub Data Set:");
        for(Instances instances : subDataSet)
        {
            System.out.println(instances.toString());
        }

        double [] entropies = new double[attribute.numValues()];
        for(int i=0; i<attribute.numValues(); i++)
        {
            entropies[i] = computeEntropy(subDataSet[i]);
        }

        System.out.println("==========entropies");
        for(double d : entropies)
        {
            System.out.println(d);
        }

        double infoGain = initialEntropi;
        for(int i=0; i<attribute.numValues(); i++)
        {
            infoGain = infoGain - ((double)subDataSet[i].numInstances()/(double)dataSet.numInstances()*entropies[i]);
        }

        return infoGain;
    }

    /**
     * Compute the entropy from a given data set
     * @param subDataSet Data set whose entropi to be computed
     * @return entropy from the data set
     */
    private double computeEntropy(Instances subDataSet)
    {
        /* Entropy = -(p1 log2 p1 + p2 log2 p2 + ...) */
        double totalInstances = subDataSet.numInstances();
        double totalClasses = subDataSet.numClasses();
        double [] classDistributions = new double[subDataSet.numClasses()];

        /* Compute each class number of instances */
        Enumeration instancesEnumeration = subDataSet.enumerateInstances();
        while(instancesEnumeration.hasMoreElements())
        {
            Instance data = (Instance) instancesEnumeration.nextElement();
            classDistributions[((int) data.classValue())]++;
        }

        /* Compute probability for each class */
        for(int i = 0; i < totalClasses; i++)
        {
            classDistributions[i] = classDistributions[i] / totalInstances;
        }

        /* compute the p * log2 p for each class */
        for(int i = 0; i < totalClasses; i++)
        {
            classDistributions[i] = classDistributions[i] * log2(classDistributions[i]);
        }

        /* compute the sum of p * log2 p */
        double sum = 0;
        for(int i=0; i<totalClasses; i++)
        {
            sum = sum + classDistributions[i];
        }

        /* return the -1 * (sum of p *log2 p) */
        /* == return -(p1 * log2 p1 + p2 * log2 p2 + ...) */
        if(sum != 0)
        {
            return -1 * sum;
        }
        else
        {
            return 0;
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

    /**
     * this method is use to divide a data set base on each value from an attribute
     * @param dataSet data Set to be splitted
     * @param attribute Attribute that is used to split
     * @return Instances that has been filtered based on the attribute's value
     */
    private Instances[] splitDataByAttribute(Instances dataSet, Attribute attribute)
    {
        /* Create several sub data set based on the possible values of an attribute */
        Instances [] subDataSet = new Instances [attribute.numValues()];
        for(int i=0; i<attribute.numValues(); i++)
        {
            subDataSet[i] = new Instances(dataSet, dataSet.numInstances());
        }

        /* Cluster each data set with the same attribute value */
        Enumeration instancesEnumeration = dataSet.enumerateInstances();
        while(instancesEnumeration.hasMoreElements())
        {
            Instance instance = (Instance) instancesEnumeration.nextElement();
            subDataSet[((int) instance.value(attribute))].add(instance);
        }

        /* Return the empty array from each data set */
        for(int i=0; i<attribute.numValues(); i++)
        {
            subDataSet[i].compactify();
        }
        return subDataSet;
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

        // only available for nominal attributes (discrete, not continuous)
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public String getRevision() {
        return super.getRevision();
    }

    public static void main (String [] args) {
        try
        {
            Classifier classifier = new MyId3();
            classifier.buildClassifier(Util.readARFF("weather.nominal.arff"));
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    private void log(String logName, String logMessage)
    {
        System.out.printf("===============[%s]===============\n%s\n===============[%s]===============\n", logName, logMessage, logName);
    }
}
