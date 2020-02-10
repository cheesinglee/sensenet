#include <vector>
#include <string>

#define NUMERIC 0
#define CATEGORICAL 1
#define IMAGE 2

using namespace std;

typedef enum PreprocessorType {
    NUMERIC = 0,
    CATEGORICAL = 1,
    IMAGE = 2,
    TEXT = 3,
} PreprocessorType;

typedef struct Preprocessor {
    PreprocessorType type;
    int nValues;
    char** values;
} Preprocessor;

Preprocessor* createPreprocessors(char* jsonSpec) {

}

void deletePreprocessors(Preprocessor* preprocessors, int nPreprocessors) {
    for (int i = 0; i < nPreprocessors; i++) {
        Preprocessor proc = preprocessors[i];
        if (proc.type == CATEGORICAL)
            deleteStringArray(proc.values, proc.nValues);
    }

    free(preprocessors);
}

typedef struct Dataset {
    int nColumns;
    int nRows;
    Preprocessor* preprocessors;
    NumericInputs* numericInputs;
    StringInputs* stringInputs;
} Dataset;

Dataset* createDataset(int nRows, int nColumns, Preprocessor* procs) {
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
        datset->stringInputs = NULL;
    }
}

void deleteDataset(Dataset* dataset) {
    if (dataset->numericInputs != NULL)
        deleteNumericInputs(dataset->numericInputs);

    if (dataset->stringInputs != NULL)
        deleteStringInputs(dataset->stringInputs);

    free(dataset);
}

int computeOffset(NumericInputs* inputs, int row, int col) {
    return input->dims[1] * row + col;
}

int computeOffset(StringInputs* inputs, int row, int col) {
    return input->dims[1] * row + col;
}

bool set(Dataset* dataset, int row, int col, float value) {
    if (row > 0 && col > 0 && row < dataset->nRows && col < dataset->nCols) {
        if (dataset->preprocessors[col].type == NUMERIC) {
            NumericInputs* inputs = dataset->numericInputs;
            int offset = computeOffset(inputs, row, col);
            dataset->numericInputs->data[offset] = value;
        }
        else return false;
    }
    else return false
}

bool set(Dataset* dataset, int row, int col, char* value) {
    if (row > 0 && col > 0 && row < dataset->nRows && col < dataset->nCols) {
        PreprocessorType type = dataset->preprocessors[col].type;

        if (type == CATEGORICAL || type == IMAGE) {
            StringInputs* inputs = dataset->stringInputs;
            int offset = computeOffset(inputs, row, col);
            inputs->data[offset] = value;
        }
        else return false;
    }
    else return false
}

bool set(Dataset* dataset, int col, float* value) {
    if (col > 0 && col < dataset->nCols) {
        PreprocessorType type = dataset->preprocessors[col].type;

        if (type == NUMERIC) {
            NumericInputs* inputs = dataset->numericInputs;
            int start = computeOffset(inputs, 0, col);
            int end = computeOffset(inputs, dataset->nRows, col);

            for (int i = start; i < end; i++) inputs->data[i] = value[i];
        }
        else return false;
    }
    else return false
}

bool set(Dataset* dataset, int col, char** value) {
    if (col > 0 && col < dataset->nCols) {
        PreprocessorType type = dataset->preprocessors[col].type;

        if (type == CATEGORICAL || type == IMAGE) {
            StringInputs* inputs = dataset->stringInputs;
            int start = computeOffset(inputs, 0, col);
            int end = computeOffset(inputs, dataset->nRows, col);

            for (int i = start; i < end; i++) inputs->data[i] = value[i];
        }
        else return false;
    }
    else return false
}
