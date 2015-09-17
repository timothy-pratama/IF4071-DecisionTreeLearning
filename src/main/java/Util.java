import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;

/**
 * Created by timothy.pratama on 16-Sep-15.
 */
public class Util {

    private static String pathDataSet = "dataSet/";
    public static enum ClassifierType
    {
        NaiveBayes,
        ID3,
        J48
    }


    /**
     * Fungsi ini digunakan untuk membaca data set dengan format arff
     * @param namaFile Nama file data set dengan format arff
     * @return Instances dari data set
     */
    public static Instances readARFF(String namaFile)
    {
        try
        {
            Instances dataSet;
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(pathDataSet + namaFile);
            dataSet = dataSource.getDataSet();
            if(dataSet.classIndex() == -1)
            {
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        }

        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Fungsi ini digunakan untuk membaca data set dalam format csv
     * @param namaFile Nama file data set dengan format csv
     * @return Instances dari data set
     */
    public static Instances readCSV(String namaFile)
    {
        try
        {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(pathDataSet + namaFile));
            Instances dataSet = csvLoader.getDataSet();
            if(dataSet.classIndex() == -1)
            {
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        }

        catch (IOException e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Fungsi ini digunakan untuk membuang atribut ke X dari sebuah data set
     * @param dataSet Data set yang akan di-filter
     * @param attributeIndex Atribut yang akan dibuang dari data set (bernilai dari 1 sampai jumlah atribut)
     * @return Data set dengan atribut ke attributeIndex sudah dibuang
     */
    public static Instances removeAttribute(Instances dataSet, int attributeIndex)
    {
        try
        {
            String options[] = new String[2];
            options[0] = "-R";
            options[1] = String.valueOf(attributeIndex);
            Remove remove = new Remove();
            remove.setOptions(options);
            remove.setInputFormat(dataSet);
            Instances newDataSet = Filter.useFilter(dataSet,remove);
            return newDataSet;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Fungsi ini digunakan untuk melakukan resampling pada data set
     * @param dataSet Data set yang akan di-resampling
     * @return data set yang sudah di-resampling
     */
    public static Instances resampleDataSet(Instances dataSet)
    {
        try
        {
            Resample resample = new Resample();
            String filterOptions = "-B 0.0 -S 1 -Z 100.0";
            resample.setOptions(Utils.splitOptions(filterOptions));
            resample.setRandomSeed((int)System.currentTimeMillis());
            resample.setInputFormat(dataSet);
            Instances newDataSet = Filter.useFilter(dataSet,resample);
            return newDataSet;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Build a classifier
     * @param classifierType Classifier Type (ID3 | J48 | NaiveBayes)
     * @return Classifier yang sesuai
     */
    public static Classifier buildClassifier(Instances dataSet, ClassifierType classifierType)
    {
        switch (classifierType)
        {
            case ID3:
            {
                try
                {
                    Id3 id3 = new Id3();
                    id3.buildClassifier(dataSet);
                    return id3;
                }

                catch (Exception e)
                {
                    e.printStackTrace();
                }
                break;
            }

            case J48:
            {
                try
                {
                    String options = "-C 0.25 -M 2";
                    J48 j48 = new J48();
                    j48.setOptions(Utils.splitOptions(options));
                    j48.buildClassifier(dataSet);
                    return j48;
                }

                catch (Exception e)
                {
                    e.printStackTrace();
                }
                break;
            }

            case NaiveBayes:
            {
                try
                {
                    NaiveBayes naiveBayes = new NaiveBayes();
                    naiveBayes.buildClassifier(dataSet);
                    return naiveBayes;
                }

                catch (Exception e)
                {
                    e.printStackTrace();
                }
                break;
            }
        }
        return null;
    }

    /**
     * Fungsi ini digunakan untuk menguji model terhadap test set
     * @param classifier Classifier yang akan diuji
     * @param dataSet Data yang digunakan untuk membuat Classifier
     * @param testSet Data yang digunakan untuk menguji Classifier
     * @return Hasil evaluasi model
     */
    public static Evaluation testClassifier(Classifier classifier, Instances dataSet, Instances testSet)
    {
        try
        {
            Evaluation evaluation = new Evaluation(dataSet);
            evaluation.evaluateModel(classifier, testSet);
            return evaluation;
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String [] args)
    {
        /* Testing reading data set from ARFF */
        System.out.println("========== Reading File From ARFF ==========");
        Instances dataSet = Util.readARFF("weather.nominal.arff");
        System.out.println(dataSet.toString());
        System.out.println("Class Attribute: " + dataSet.attribute(dataSet.classIndex()));

        /* Testing resampling data set */
        System.out.println("\n========== Resampling Data Set ==========");
        dataSet = Util.resampleDataSet(dataSet);
        System.out.println(dataSet.toString());

        /* Testing reading data set from CSV */
        System.out.println("\n========== Reading File From CSV ==========");
        dataSet = Util.readCSV("weather.nominal.csv");
        System.out.println(dataSet.toString());
        System.out.println("Class Attribute: " + dataSet.attribute(dataSet.classIndex()));

        /* Testing removing an attribute from data set */
        System.out.println("\n========== Removing Class Attributes ==========");
        dataSet = readARFF("weather.nominal.arff");
        dataSet = Util.removeAttribute(dataSet,dataSet.numAttributes());
        System.out.println(dataSet.toString());

        /* Testing building classifier */
        System.out.println("\n========== Building Naive Bayes Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        Classifier classifier = Util.buildClassifier(dataSet, ClassifierType.NaiveBayes);
        System.out.println(classifier.toString());

        System.out.println("\n========== Building ID3 Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        classifier = Util.buildClassifier(dataSet, ClassifierType.ID3);
        System.out.println(classifier.toString());

        System.out.println("\n========== Building J48 Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        classifier = Util.buildClassifier(dataSet, ClassifierType.J48);
        System.out.println(classifier.toString());

        System.out.println("\n========== Testing Naive Bayes Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        Instances trainSet = readARFF("weather.nominal.test.arff");
        classifier = Util.buildClassifier(dataSet, ClassifierType.NaiveBayes);
        Evaluation eval = Util.testClassifier(classifier, dataSet, trainSet);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Testing ID3 Classifier ==========");
        classifier = Util.buildClassifier(dataSet, ClassifierType.ID3);
        eval = Util.testClassifier(classifier, dataSet, trainSet);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Testing J48 Classifier ==========");
        classifier = Util.buildClassifier(dataSet, ClassifierType.J48);
        eval = Util.testClassifier(classifier, dataSet, trainSet);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));
    }
}
