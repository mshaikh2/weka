import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.converters.CSVLoader;

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class hierClu {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
	
	public static void main(String[] args) throws Exception
	{
		SimpleKMeans kmeans = new SimpleKMeans();
		 
		kmeans.setSeed(10);
 
		//important parameter to set: preserver order, number of cluster.
		kmeans.setPreserveInstancesOrder(true);
		kmeans.setNumClusters(2);
 
		BufferedReader datafile = readDataFile("TrainingData\\diabetes.arff"); 
		Instances data = new Instances(datafile);
 
		kmeans.buildClusterer(data);
 
		// This array returns the cluster number (starting with 0) for each instance
		// The array has as many elements as the number of instances
		int[] assignments = kmeans.getAssignments();
		int i=0;
		for(int clusterNum : assignments) {
			System.out.println(clusterNum + "======" + data.get(i));
		    i++;
		}
		Instances instances = kmeans.getClusterCentroids();
		System.out.println(instances);
		
		//Manhattan - true
		//Euclidean - false
		//--------------------------------------------------------------
		HierarchicalClusterer hc = new HierarchicalClusterer(true);
		hc.buildClusterer(data);
		System.out.println("Manhattan: " + hc.graph());
		//--------------------------------------------------------------
		HierarchicalClusterer hc1 = new HierarchicalClusterer(false);
		hc1.buildClusterer(data);
		System.out.println("Euclidean: " + hc1.graph());
		//--------------------------------------------------------------
		
	}
}