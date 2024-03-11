/********************************
Name: Christian Johnson
Username: N/A
Problem Set: PS4
Due Date: August 8th, 2023
********************************/

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

@SuppressWarnings("static-access")
public class PS4 {

	public static void main(String args[]) throws IOException {
		String w1 = args[0];
		String w2 = args[1];
		String xdata = args[2];
		String ydata = args[3];
		double lr = 0.25;
		double lambda = 2.0;
		Matrix matrix = new Matrix(); // object to manipulate matrices

		// load data
		double[][] w1Matrix = loadMatrix(w1, 30, 785);
		double[][] w2Matrix = loadMatrix(w2, 10, 31);
		double[][] xMatrix = addBiasCol(loadMatrix(xdata, 10000, 784));
		double[][] yMatrix = oneHotEncode(loadMatrix(ydata, 10000, 1));
		double[][] yhat = null; // for printing predicted vs true after convergence

		System.out.println("*************************************************************************");
		System.out.println("Problem Set:  Problem Set 4:  Neural Network");
		System.out.println("Name:         Christian Johnson");
		System.out.println("Syntax:       java PS4  " + w1 + " " + w2 + " " + xdata + " " + ydata);
		System.out.println("*************************************************************************");
		System.out.println();
		System.out.println("Training Phase: " + xdata);
		System.out.println("--------------------------------------------------------------");
		System.out.println("   => Number of Entries (n):      "  + xMatrix.length);
		System.out.println("   => Number of Features (p):     "  + (xMatrix[0].length - 1));
		System.out.println();
		System.out.println("Starting Gradient Descent: ");
		System.out.println("--------------------------------------------------------------");
		System.out.println();
		
		BufferedWriter bw = new BufferedWriter(new FileWriter("chart.txt"));
		int epoch = 1;
		while (epoch <= 700 ) {
			System.out.printf("Epoch %3d: ", epoch);
		// Forward Phase
			double[][] H1 = addBiasCol(sigmoidActivation(matrix.multiply(xMatrix, matrix.transpose(w1Matrix))));
			yhat = sigmoidActivation(matrix.multiply(H1, matrix.transpose(w2Matrix)));

		// Backward Phase
			// get deltas
				// delta2 = yhat - y
			double[][] delta2 = matrixSubtraction(yhat, yMatrix); 
				// delta1 = (delta2 x w2*) * ((xMatrix x t(w1*)) * (1 - (xMatrix x t(w1*))))
			double[][] delta1 = delta1(delta2, w2Matrix, w1Matrix, xMatrix);
			// get gradient of loss
				// gradW2 = (1/n)(t(delta2) x H1) + (lambda/n)(sum(w2*)
			double[][] gradW2 = matrixPlusDouble(gradient(delta2, H1), gradientRegularization(w2Matrix, lambda));
				// gradW1 =  (1/n)(t(delta1) x xMatrix) + (lambda/n)(sum(w1*)
			double[][] gradW1 = matrixPlusDouble(gradient(delta1, xMatrix), gradientRegularization(w1Matrix, lambda));
			// update weights
				// w2New = w2 - (lr * gradW2)
			w2Matrix = matrixSubtraction(w2Matrix, matrixTimesDouble(gradW2, lr));
				// w1New = w1 - (lr * gradW1)
			w1Matrix = matrixSubtraction(w1Matrix, matrixTimesDouble(gradW1, lr));

			double loss = loss(yMatrix, yhat, w1Matrix, w2Matrix, lambda);
			System.out.printf("Loss of %-6.3f \n", loss);
			bw.write(epoch + ", " + loss + "\n");
			epoch++;
		}
		
		bw.close();
		System.out.println();
		System.out.printf("Epochs Required:  %-3d \n", (epoch - 1));
		System.out.println();
		System.out.println("Testing Phase (first 10 records): ");
		System.out.println("--------------------------------------------------------------");
		test(yMatrix, yhat);
		
		// save weights to file
		saveWeights(w1Matrix, "w1out.txt");
		saveWeights(w2Matrix, "w2out.txt");
	}
	
	public static double[][] sigmoidActivation(double[][] hl) {
		double[][] newMatrix = new double[hl.length][hl[0].length];
		for (int r = 0; r < hl.length; r++) {
			for (int c = 0; c < hl[0].length; c++) {
				newMatrix[r][c] = (1.0 / (1.0 + Math.exp(-(hl[r][c]))));
			}
		}
		return newMatrix;
	}
	
	public static double[][] delta1(double[][] delta, double[][] weights2, double[][] weights1, double[][] xMatrix) {
		Matrix matrix = new Matrix();
		weights2 = matrix.dropFirstColumn(weights2);
		weights1 = matrix.dropFirstColumn(weights1);
		xMatrix = matrix.dropFirstColumn(xMatrix);
		double[][] hl = sigmoidActivation(matrix.multiply(xMatrix, matrix.transpose(weights1)));
		double[][] deriveSigmoid = elementWiseMultiply(hl, matrixMinusDouble(hl, 1.0));
		double[][] delta1 = (elementWiseMultiply(matrix.multiply(delta, weights2), deriveSigmoid));
		return delta1;
	}
	
	public static double[][] gradient(double[][] delta, double[][] hl) {
		Matrix matrix = new Matrix();
		double[][] tDeltaH1 = matrix.multiply(matrix.transpose(delta), hl);
		return matrixTimesDouble(tDeltaH1, (1.0 / 10000));
	}
	
	public static double gradientRegularization(double[][] a, double i) {
		double total = 0.0;
		for (int r = 0; r < a.length; r++) {
			for (int c = 1; c < a[0].length; c++) {
				total += (i / 10000) * a[r][c];
			}
		}
		return total;
	}

	public static double loss(double[][] y, double[][] yhat, double[][] weights1, double[][] weights2, double lambda) {
		double loss = 0.0;
		double reg = lossRegularization(weights1, weights2);
		reg = (lambda / (2.0 * y.length)) * reg;
		for (int r = 0; r < y.length; r++) {
			for (int c = 0; c < y[0].length; c++) {
				loss += (((-1.0 * y[r][c]) * Math.log(yhat[r][c])) - ((1.0 - y[r][c]) * Math.log(1.0 - yhat[r][c])));
			}
		}

		loss = ((1.0 / y.length) * loss) + (reg);

		return loss;
	}

	public static double lossRegularization(double[][] weights1, double[][] weights2) {
		double total = 0;
		for (int r = 0; r < weights1.length; r++) {
			for (int c = 0; c < weights1[0].length; c++) {
				total += Math.pow(weights1[r][c], 2.0);
			}
		}

		for (int r = 0; r < weights2.length; r++) {
			for (int c = 0; c < weights2[0].length; c++) {
				total += Math.pow(weights2[r][c], 2.0);
			}
		}

		return total;
	}

	public static void accuracy(double[][] prediction, double[][] trueY) {
		double numCorrect = 0.0;
		double total = prediction.length;
		for (int r = 0; r < prediction.length; r++) {
			int trueVal = 0;
			int colMax = 0;
			double predMax = 0;
			for (int c = 0; c < prediction[0].length; c++) {
				if (predMax < prediction[r][c]) {
					predMax = prediction[r][c];
					colMax = c;
				}
				if (trueY[r][c] == 1) {
					trueVal = c;
				}
			}
			if (trueVal == colMax) {
				numCorrect++;
			}

		}
		double accuracy = (numCorrect / total) * 100;
		System.out.printf("  Accuracy: %4.2f%% \n", accuracy);
	}
	
	public static void test(double[][] y, double[][] yhat) {
		double numCorrect = 0.0;
		for (int r = 0; r < 10; r++) {
			boolean isCorrect = false;
			int trueVal = 0;
			int colMax = 0;
			double predMax = 0;
			for (int c = 0; c < yhat[0].length; c++) {
				if (predMax < yhat[r][c]) {
					predMax = yhat[r][c];
					colMax = c;
				}
				if (y[r][c] == 1) {
					trueVal = c;
				}
			}
			if (trueVal == colMax) {
				numCorrect++;
				isCorrect = true;
			}
			
			if(colMax == 0) {
				colMax = 10;
			}
			
			if(trueVal == 0) {
				trueVal = 10;
			}
			System.out.printf("   Test Record %-2d:  %-2d    Prediction:  %2d   Correct: %s \n", (r + 1), trueVal, colMax, isCorrect);

		}
		double accuracy = (numCorrect / 10) * 100;
		System.out.println();
		System.out.printf("   => Number of Test Entries (n): %-2d \n", 10);
		System.out.printf("   => Accuracy:                   %2.2f%% \n", accuracy);
	}
	

// MATRIX FUNCTIONS --------------------------------------------------------------------------
	public static void printMatrix(double[][] matrix) {
		for (int r = 0; r < 5; r++) {
			for (int c = 0; c < matrix[0].length; c++) {
				System.out.printf("%5.3f   ", matrix[r][c]);
			}
			System.out.println();
		}
	}

	public static double[][] loadMatrix(String filename, int rowNum, int colNum) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line;
		double[][] matrix = new double[rowNum][colNum];
		int row = 0;
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");
			for (int col = 0; col < tokens.length; col++) {
				matrix[row][col] = Double.parseDouble(tokens[col]);
			}
			row++;
		}
		br.close();
		return matrix;
	}
	
	public static void saveWeights(double[][] weights, String name) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(name));
		
		for(int r = 0; r < weights.length; r++) {
			for(int c = 0; c < weights[r].length; c++) {
				if(c+1 != weights[r].length) {
					bw.write(String.format("%7.4f, ",weights[r][c]));
				} else {
					bw.write(String.format("%7.4f ",weights[r][c]));
				}
			}
			bw.write("\n");
		}
		
		bw.close();
	}
	
	public static double[][] addBiasCol(double[][] matrix) {
		double[][] newMatrix = new double[matrix.length][matrix[0].length + 1];
		for (int r = 0; r < matrix.length; r++) {
			newMatrix[r][0] = 1.0;
			for (int c = 1; c < matrix[0].length; c++) {
				newMatrix[r][c] = matrix[r][c - 1];
			}
		}
		return newMatrix;
	}

	public static double[][] oneHotEncode(double[][] a) {
		double[][] newMatrix = new double[a.length][10];
		for (int r = 0; r < a.length; r++) {
			for (int c = 0; c < newMatrix[0].length; c++) {
				int num = (int) a[r][0];
				if (c == num) {
					newMatrix[r][c] = 1.0;
				}
				if (num == 10 && c == 0) {
					newMatrix[r][c] = 1.0;
				}
				if (c != num && newMatrix[r][c] != 1) {
					newMatrix[r][c] = 0.0;
				}
			}
		}
		return newMatrix;
	}
	
	public static double[][] elementWiseMultiply(double[][] a, double[][] b) {
		double[][] newMatrix = new double[a.length][a[0].length];
		for (int r = 0; r < a.length; r++) {
			for (int c = 0; c < a[0].length; c++) {
				newMatrix[r][c] = a[r][c] * b[r][c];
			}
		}
		return newMatrix;
	}
	
	public static double[][] matrixSubtraction(double[][] a, double[][] b) {
		double[][] newMatrix = new double[a.length][a[0].length];
		for (int r = 0; r < a.length; r++) {
			for (int c = 0; c < a[0].length; c++) {
				newMatrix[r][c] = a[r][c] - b[r][c];
			}
		}
		return newMatrix;
	}

	public static double[][] matrixTimesDouble(double[][] a, double i) {
		double[][] newMatrix = new double[a.length][a[0].length];
		for (int r = 0; r < a.length; r++) {
			for (int c = 0; c < a[0].length; c++) {
				newMatrix[r][c] = i * a[r][c];
			}
		}
		return newMatrix;
	}
	
	public static double[][] matrixPlusDouble(double[][] a, double i) {
		double[][] newMatrix = new double[a.length][a[0].length];
		for (int r = 0; r < a.length; r++) {
			for (int c = 0; c < a[0].length; c++) {
				newMatrix[r][c] = a[r][c] + i;
			}
		}
		return newMatrix;
	}

	public static double[][] matrixMinusDouble(double[][] a, double i) {
		double[][] newMatrix = new double[a.length][a[0].length];
		for (int r = 0; r < a.length; r++) {
			for (int c = 0; c < a[0].length; c++) {
				newMatrix[r][c] = i - a[r][c];
			}
		}
		return newMatrix;
	}

}
