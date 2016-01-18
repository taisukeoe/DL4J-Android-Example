package org.deeplearning4j.androidexamples;

import android.content.Context;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.io.IOException;

/**
 * Created by taisukeoe on 16/01/18.
 */
public class AndroidMnistDataSetIterator extends BaseDatasetIterator {
    /**Get the specified number of examples for the MNIST training data set.
     * @param batch the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param binarize whether to binarize mnist or not
     * @throws IOException
     */
    public AndroidMnistDataSetIterator(Context context,int batch, int numExamples, boolean binarize) throws IOException {
        this(context, batch,numExamples,binarize,true,false,0);
    }

    /** Constructor to get the full MNIST data set (either test or train sets) without binarization (i.e., just normalization
     * into range of 0 to 1), with shuffling based on a random seed.
     * @param batchSize
     * @param train
     * @throws IOException
     */
    public AndroidMnistDataSetIterator(Context context, int batchSize, boolean train, int seed) throws IOException{
        this(context, batchSize, (train ? MnistDataFetcher.NUM_EXAMPLES : MnistDataFetcher.NUM_EXAMPLES_TEST), false, train, true, seed);
    }

    /**Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
     * @param batch Size of each patch
     * @param numExamples total number of examples to load
     * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
     * @param train Train vs. test set
     * @param shuffle whether to shuffle the examples
     * @param rngSeed random number generator seed to use when shuffling examples
     */
    public AndroidMnistDataSetIterator(Context context,int batch, int numExamples, boolean binarize, boolean train, boolean shuffle, long rngSeed) throws IOException {
        super(batch, numExamples,new AndroidMnistDataFetcher(context,binarize,train,shuffle,rngSeed));
    }
}
