import java.io.BufferedReader;
import java.io.FileReader;

import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;
public class hierClu {
	public static void main(String[] args) throws Exception
	{
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("TrainingData\\diabetes.arff"));
		Instances train = new Instances(breader);
		breader.close();
		
		HierarchicalClusterer hc = new HierarchicalClusterer();
		hc.buildClusterer(train);
		System.out.println(hc.graph());
	}
}
