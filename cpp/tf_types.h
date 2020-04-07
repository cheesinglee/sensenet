#include <stdlib.h>
#include <tensorflow/c/c_api.h>

int dataLength(int64_t* dims, int nDims) {
    if (nDims > 0) {
        int dLen = 1;
        for (int i = 0; i < nDims; i++) dLen *= dims[i];

        return dLen;
    }
    else return 0;
}

void deleteStringArray(char** strings, int nStrings) {
    for (int i = 0; i < nStrings; i++)
        if (strings[i] != NULL) free(strings[i]);
}

typedef enum DataType {
    NUMERIC = 0,
    CATEGORICAL = 1,
    IMAGE = 2,
    TEXT = 3,
} DataType;

DataType* createDataTypes(const char* filePath) {
    // TODO: Actually do this
    return NULL;
}

typedef struct ModelSummary {
    TF_Session* session;
    TF_Graph* graph;
    TF_Output* feeds;
    TF_Output* fetches;
    DataType* inputTypes;
    int numericInputs;
    int stringInputs;
    int totalInputs;
} Model;

void deleteModel(Model* model) {
    TF_Status* status = TF_NewStatus();

    TF_DeleteSession(model->session, status);
    TF_DeleteGraph(model->graph);
    TF_DeleteStatus(status);

    if (model->feeds != NULL) free(model->feeds);
    if (model->fetches != NULL) free(model->fetches);
    if (model->inputTypes != NULL) free(model->inputTypes);

    free(model);
}

typedef struct ModelNumericOutputs {
    float* data;
    int nDims;
    int64_t* dims;
    char* statusMessage;
    int statusCode;
} Outputs;

void deleteOutputs(Outputs* output) {
    if (output->data != NULL) free(output->data);
    if (output->statusMessage != NULL) free(output->statusMessage);
    if (output->dims != NULL) free(output->dims);

    free(output);
}

typedef struct ModelNumericInputs {
    float* data;
    int nDims;
    int64_t* dims;
} NumericInputs;

NumericInputs* createNumericInputs(float* data, int nDims, int64_t* dims) {
    NumericInputs* input = malloc(sizeof(NumericInputs));

    input->data = data;
    input->nDims = nDims;
    input->dims = dims;

    return input;
}

void deleteNumericInputs(NumericInputs* input) {
    if (input->data != NULL) free(input->data);
    if (input->dims != NULL) free(input->dims);
    free(input);
}

typedef struct ModelStringInputs {
    char** data;
    int nDims;
    int64_t* dims;
} StringInputs;

StringInputs* createStringInputs(char** data, int nDims, int64_t* dims) {
    StringInputs* input = malloc(sizeof(StringInputs));
    return NULL;
    // TODO: Actually do this
}

void deleteStringInputs(StringInputs* input) {
    if (input->nDims > 0) {
        int nStrings = dataLength(input->dims, input->nDims);
        deleteStringArray(input->data, nStrings);

        if (input->data != NULL) free(input->data);
        if (input->dims != NULL) free(input->dims);
    }

    free(input);
}
