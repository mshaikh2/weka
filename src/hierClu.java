import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.clusterers.HierarchicalClusterer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
public class hierClu {
	public static void main(String[] args) throws Exception
	{
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("TrainingData\\diabetes.csv"));
		System.out.println(breader);
		Instances train = new Instances(breader);
		breader.close();
		//Manhattan - true
		//Euclidean - false
		//--------------------------------------------------------------
		HierarchicalClusterer hc = new HierarchicalClusterer(true);
		hc.buildClusterer(train);
		System.out.println("Manhattan: " + hc.graph());
		//--------------------------------------------------------------
		HierarchicalClusterer hc1 = new HierarchicalClusterer(false);
		hc1.buildClusterer(train);
		System.out.println("Euclidean: " + hc1.graph());
		//--------------------------------------------------------------
		/*double[] l = new double[train.numAttributes()];
		for(int i=0;i<train.numAttributes();i++){
			l[i] = 0;
		}
		Instance inst = new DenseInstance(1, l);
		hc.getDistanceFromZero(inst);*/
		
		
	}
}