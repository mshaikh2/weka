import java.io.*;
import weka.classifiers.trees.J48;
import weka.core.*;
public class wk {
	public static void main(String[] args) throws Exception
	{
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("C:\\Users\\Mihir\\Documents\\Weka-3-8\\data\\iris.arff"));
		
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes() - 1);
		
		breader = new BufferedReader(new FileReader("C:\\Users\\Mihir\\Documents\\Weka-3-8\\data\\iris1.arff"));
		Instances test = new Instances(breader);
		test.setClassIndex(train.numAttributes() - 1);
		
		breader.close();
		
		J48 tree = new J48();
		tree.buildClassifier(train);
		Instances labeled = new Instances(test);
		
		for (int i = 0; i < test.numInstances(); i++)
		{
			double clsLabel = tree.classifyInstance(test.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
		}
		
		BufferedWriter writer = new BufferedWriter(
				new FileWriter("C:\\Users\\Mihir\\Documents\\Weka-3-8\\data\\labeled.arff"));
		writer.write(labeled.toString());
		
		writer.close();
	}
}
