package com.Cascadeexample;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.web.HttpMessageConverters;
import org.springframework.context.annotation.Bean;
import org.springframework.http.converter.ByteArrayHttpMessageConverter;

@SpringBootApplication
public class Main {
	 public static String ROOT = "upload-dir";
	public static void main(String[] args) {
		SpringApplication.run(Main.class, args);

	}
	}