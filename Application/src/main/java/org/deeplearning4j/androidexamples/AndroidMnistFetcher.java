package org.deeplearning4j.androidexamples;

import android.content.Context;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Created by taisukeoe on 16/01/18.
 */

public class AndroidMnistFetcher extends MnistFetcher {
//    private File fileDir;
//    private static final String trainingFilesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
//    private static final String trainingFilesFilename = "images-idx3-ubyte.gz";
//    public static final String trainingFilesFilename_unzipped = "images-idx3-ubyte";
//    private static final String trainingFileLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
//    private static final String trainingFileLabelsFilename = "labels-idx1-ubyte.gz";
//    public static final String trainingFileLabelsFilename_unzipped = "labels-idx1-ubyte";
//
//    //Test data:
//    private static final String testFilesURL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
//    private static final String testFilesFilename = "t10k-images-idx3-ubyte.gz";
//    public static final String testFilesFilename_unzipped = "t10k-images-idx3-ubyte";
//    private static final String testFileLabelsURL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
//    private static final String testFileLabelsFilename = "t10k-labels-idx1-ubyte.gz";
//    public static final String testFileLabelsFilename_unzipped = "t10k-labels-idx1-ubyte";

    public AndroidMnistFetcher(Context context) {
        BASE_DIR = context.getCacheDir();
        if (!BASE_DIR.exists())
            BASE_DIR.mkdir();
        FILE_DIR = new File(BASE_DIR, LOCAL_DIR_NAME);
        if (!FILE_DIR.exists())
            FILE_DIR.mkdir();
    }

//    public File downloadAndUntar() throws IOException {
//        if (fileDir != null) {
//            return fileDir;
//        }
//        // mac gives unique tmp each run and we want to store this persist
//        // this data across restarts
//        File tmpDir = new File(System.getProperty("user.home"));
//
//        File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
//        if (!(baseDir.isDirectory() || baseDir.mkdir())) {
//            throw new IOException("Could not mkdir " + baseDir);
//        }
//
//
//        log.info("Downloading mnist...");
//        // getFromOrigin training records
//        File tarFile = new File(baseDir, trainingFilesFilename);
//        File tarFileLabels = new File(baseDir, testFilesFilename);
//
//        if (!tarFile.isFile()) {
//            FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);
//        }
//
//        if (!tarFileLabels.isFile()) {
//            FileUtils.copyURLToFile(new URL(testFilesURL), tarFileLabels);
//        }
//
//        ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), baseDir.getAbsolutePath());
//        ArchiveUtils.unzipFileTo(tarFileLabels.getAbsolutePath(), baseDir.getAbsolutePath());
//
//        // getFromOrigin training records
//        File labels = new File(baseDir, trainingFileLabelsFilename);
//        File labelsTest = new File(baseDir, testFileLabelsFilename);
//
//        if (!labels.isFile()) {
//            FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), labels);
//        }
//        if (!labelsTest.isFile()) {
//            FileUtils.copyURLToFile(new URL(testFileLabelsURL), labelsTest);
//        }
//
//        ArchiveUtils.unzipFileTo(labels.getAbsolutePath(), baseDir.getAbsolutePath());
//        ArchiveUtils.unzipFileTo(labelsTest.getAbsolutePath(), baseDir.getAbsolutePath());
//
//        fileDir = baseDir;
//        return fileDir;
//    }
}
