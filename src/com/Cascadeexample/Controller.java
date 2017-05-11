package com.Cascadeexample;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class Controller {
	@RequestMapping(value="/upload")
	public String ImageApp()
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat query = Imgcodecs.imread("umee.jpg");
		Mat overlay=Imgcodecs.imread("fedora.png");
			try {
				
				MainClass.loadCascade();
				//Functions.display(Converter.mat2Img(MainClass.detectAndDrawFace(query, overlay)));
				Mat output=MainClass.detectAndDrawFace(query, overlay);
				
				BufferedImage img=Converter.mat2Img(output);		
				File temp=new File("umairTopi.jpg");
				ImageIO.write(img,"jpg",temp);
				System.out.println("Image Saved In Directory..");
				
				
		}
			catch (Exception e) {
					System.err.println(e.getMessage());
	}
		
		
	//	MainClass.detectAndDrawFace(query, overlay);
		 
		 return "Image Saved";
	}
	
	@RequestMapping(value="download", headers = "Accept=image/jpeg")
	 public @ResponseBody byte[] doDownload() throws IOException  {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	 try {
        BufferedImage bufferedImage = ImageIO.read(new File("zubair.jpg"));
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ImageIO.write( bufferedImage  , "jpg", byteArrayOutputStream);
        return byteArrayOutputStream.toByteArray();

    } catch (Exception e) {
        throw new RuntimeException(e);
       
			}
		}
	}
