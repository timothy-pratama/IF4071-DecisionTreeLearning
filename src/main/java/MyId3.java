import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by timothy.pratama on 21-Sep-15.
 */
public class MyId3 extends Classifier {
    @Override
    public void buildClassifier(Instances instances) throws Exception {

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
        return super.getCapabilities();
    }

    @Override
    public String getRevision() {
        return super.getRevision();
    }
}
