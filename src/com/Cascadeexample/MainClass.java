package com.Cascadeexample;

import java.io.IOException;
import java.io.ObjectInputStream.GetField;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.objdetect.CascadeClassifier;

public class MainClass {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private static CascadeClassifier faceDetector;
	

	public static void main(String[] args) throws IOException {
		// Calling method
		Mat query = Imgcodecs.imread("queryimg2.jpg");
//	    Mat query = Imgcodecs.imread("query.jpg");
		Mat overlay = Imgcodecs.imread("fedora.png");
		loadCascade();
		try {

			Functions.display(Converter.mat2Img(detectAndDrawFace(query, overlay)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		;
	}

	public static void loadCascade() {
		String cascadePath = "lbpcascade_frontalface.xml";
		faceDetector = new CascadeClassifier(cascadePath);
	}

	public static Mat detectAndDrawFace(Mat image, Mat overlay) {
		MatOfRect faceDetections = new MatOfRect();
		faceDetector.detectMultiScale(image, faceDetections);

		// Draw a bounding box around each face.

		for (Rect rect : faceDetections.toArray()) {
			
//			Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(
//					rect.x + rect.width, rect.y + rect.height), new Scalar(
//					0, 255,0));

			  double hatGrowthFactor = 2.3;//1.8;   
		        
		        int hatWidth = (int) (rect.width *hatGrowthFactor);
		        int hatHeight = (int) (hatWidth * overlay.height() / overlay.width());
		        int roiX =  rect.x - (hatWidth-rect.width)/2;
		        int roiY =  (int) (rect.y  - 0.6*hatHeight);
		        roiX =  roiX<0?0:roiX;
		        roiY = roiY<0?0:roiY;
		        hatWidth = hatWidth+roiX > image.width() ? image.width() -roiX : hatWidth;
		        
		        hatHeight = hatHeight+roiY > image.height() ? image.height() - roiY : hatHeight;
		        Rect roi = new Rect( new Point(roiX,roiY), new Size( hatWidth, hatHeight));		        
		        
		        Mat resized = new Mat();
		        Size size = new Size(hatWidth,hatHeight);
		        Imgproc.resize(overlay,resized, size);
		        Mat destinationROI = image.submat( roi );
		        resized.copyTo( destinationROI , resized);

		      //  break;

		}
		return image;

	}
}
