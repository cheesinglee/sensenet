#include <stdlib.h>

#include "tf_model.h"

typedef struct Dataset {
    int nColumns;
    int nRows;
    DataType* types;
    NumericInputs* numericInputs;
    StringInputs* stringInputs;
} Dataset;

Dataset* createDataset(int nRows, int nColumns, DataType* procs) {
    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->nColumns = nColumns;
    dataset->nRows = nRows;

    int nStrings = 0;

    for (int i = 0; i < nColumns; i++)
        if (procs[i].type != NUMERIC) nStrings++;

    int64_t* numDims = malloc(sizeof(int64_t) * 2);
    numDims[0] = nRows;
    numDims[1] = nColumns;

    float* numData = malloc(sizeof(float) * nRows * nColumns);
    dataset->numericInputs = createNumericInputs(numData, 2, numDims);

    if (nStrings > 0) {
        int64_t* strDims = malloc(sizeof(int64_t) * 2);
        strDims[0] = nRows;
        strDims[1] = nStrings;

        char** strData = malloc(sizeof(char*) * nRows * nStrings);
        dataset->stringInputs = createStringInputs(strData, 2, numDims);
    }
    else {
        dataset->stringInputs = NULL;
    }

    return dataset;
}

void deleteDataset(Dataset* dataset) {
    if (dataset->numericInputs != NULL)
        deleteNumericInputs(dataset->numericInputs);

    if (dataset->stringInputs != NULL)
        deleteStringInputs(dataset->stringInputs);

    free(dataset);
}

int computeNumericOffset(NumericInputs* inputs, int row, int col) {
    return inputs->dims[1] * row + col;
}

int computeStringOffset(StringInputs* inputs, int row, int col) {
    return inputs->dims[1] * row + col;
}

bool setNumeric(Dataset* dataset, int row, int col, float value) {
    if (row > 0 && col > 0 && row < dataset->nRows && col < dataset->nColumns) {
        DataType type = dataset->types[col];

        if (type == NUMERIC) {
            NumericInputs* inputs = dataset->numericInputs;
            int offset = computeNumericOffset(inputs, row, col);
            dataset->numericInputs->data[offset] = value;

            return true;
        }
        else return false;
    }
    else return false;
}

bool setString(Dataset* dataset, int row, int col, char* value) {
    if (row > 0 && col > 0 && row < dataset->nRows && col < dataset->nColumns) {
        DataType type = dataset->types[col];

        if (type == CATEGORICAL || type == IMAGE) {
            StringInputs* inputs = dataset->stringInputs;
            int offset = computeStringOffset(inputs, row, col);
            inputs->data[offset] = value;

            return true;
        }
        else return false;
    }
    else return false;
}

bool setNumerics(Dataset* dataset, int col, float* value) {
    if (col > 0 && col < dataset->nColumns) {
        DataType type = dataset->types[col];

        if (type == NUMERIC) {
            NumericInputs* inputs = dataset->numericInputs;
            int start = computeNumericOffset(inputs, 0, col);
            int end = computeNumericOffset(inputs, dataset->nRows, col);

            for (int i = start; i < end; i++) inputs->data[i] = value[i];
            return true;
        }
        else return false;
    }
    else return false;
}

bool setStrings(Dataset* dataset, int col, char** value) {
    if (col > 0 && col < dataset->nColumns) {
        DataType type = dataset->types[col];

        if (type == CATEGORICAL || type == IMAGE) {
            StringInputs* inputs = dataset->stringInputs;
            int start = computeStringOffset(inputs, 0, col);
            int end = computeStringOffset(inputs, dataset->nRows, col);

            for (int i = start; i < end; i++) inputs->data[i] = value[i];
            return true;
        }
        else return false;
    }
    else return false;
}
