#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tf_types.h"

TF_Output* makeIO(TF_Graph* graph, const char** names, int nInputs) {
    TF_Output* feeds = malloc(sizeof(TF_Input) * nInputs);
    for (int i = 0; i < nInputs; i++) {
        feeds[i].oper = TF_GraphOperationByName(graph, names[i]);
        feeds[i].index = 0;
    }

    return feeds;
}

TF_Tensor* floatTensor(NumericInputs* input) {
    float* data = input->data;
    int64_t* dims = input->dims;

    int nvals = dataLength(input->dims, input->nDims);
    int dSize = sizeof(float) * nvals;

    TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, input->dims, input->nDims, dSize);
    memcpy(TF_TensorData(t), input->data, dSize);

    return t;
}

TF_Tensor* stringTensor(StringInputs* input) {
    return NULL;
}

Outputs* parseOutputs(TF_Tensor* outTensor, TF_Status* status) {
    Outputs* output = malloc(sizeof(Outputs));

    output->statusCode = (int)TF_GetCode(status);
    output->statusMessage = NULL;
    output->nDims = 0;
    output->dims = NULL;
    output->data = NULL;

    if (output->statusCode == 0) {
        output->nDims = TF_NumDims(outTensor);
        output->dims = malloc(sizeof(int64_t) * output->nDims);

        size_t dataSize = sizeof(float);

        for (int i = 0; i < output->nDims; i++) {
            output->dims[i] = TF_Dim(outTensor, i);
            dataSize *= output->dims[i];
        }

        output->data = (float*)(malloc(dataSize));
        memcpy(output->data, TF_TensorData(outTensor), dataSize);

        TF_DeleteTensor(outTensor);
    }
    else {
        const char* tfMessage = TF_Message(status);
        output->statusMessage = malloc(sizeof(char) * strlen(tfMessage));

        strcpy(output->statusMessage, tfMessage);
    }

    return output;
}

void printArray(float* data, int offset, int len) {
    int end = offset + len;

    printf("[");
    for (int i = offset; i < end - 1; i++) {
        printf("%6.2f ", data[i]);
    }
    printf("%6.2f]", data[end - 1]);
}

// Load a TensorFlow model from a SavedModel directory
//
// The string `exportDir` is a directory in which a model has been
// preiously saved in the SavedModel format, (e.g., in python using
// tf.keras.Model.save).  The array of strings `inputNames` is a list
// of the named inputs to the model, which can be discovered with the
// `saved_model_cli` and are generally of the form
// "<exportedSignature>_<graphInputName>".  `nInputs` is of course the
// number of names.
//
// The function returns a Model struct containing pointers to the
// created TF_Graph and TF_Session and to the TF_Output* structures
// used to represent the inputs and outputs to the model, and should
// be used mainly as an input argument to `runModel` and finally to
// `deleteModel`.
Model* loadModel(const char* exportDir,
                const char** inputNames,
                int nNumeric,
                int nString) {

    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sessionOpts = TF_NewSessionOptions();

    const char* tag = "serve";
    const char* outputNames[] = {"PartitionedCall"};

    TF_Session* session = TF_LoadSessionFromSavedModel(sessionOpts,
                                                       NULL,
                                                       exportDir,
                                                       &tag,
                                                       1,
                                                       graph,
                                                       NULL,
                                                       status);


    Model* model = malloc(sizeof(Model));
    int totalInputs = nNumeric + nString;

    model->session = session;
    model->graph = graph;
    model->feeds = makeIO(graph, inputNames, totalInputs);
    model->fetches = makeIO(graph, outputNames, 1);

    model->numericInputs = nNumeric;
    model->stringInputs = nString;
    model->totalInputs = totalInputs;

    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(sessionOpts);

    return model;
}

// Runs the Model `model`, created using `loadModel`, on the given inputs.
//
// Invokes the loaded model on the given inputs.  Input tensors are
// assumed to have been created by the `createNumericInputs` and
// `createStringInputs` functions.
//
// Returns an `Outputs` structure which will contain either the
// floating point outputs of the model, in a structure similar to the
// numeric inputs, as well as a status code.  If the status code is
// non-zero, a `message` in the returned structure is populated and
// all other output information is NULL.
Outputs* runModel(Model* model, NumericInputs** numIns, StringInputs** strIns) {
    TF_Status* status = TF_NewStatus();
    TF_Tensor** feedValues = malloc(sizeof(TF_Tensor*) * model->totalInputs);
    TF_Tensor** fetchValues = malloc(sizeof(TF_Tensor*));

    for (int i = 0; i < model->numericInputs; i++)
        feedValues[i] = floatTensor(numIns[i]);

    for (int i = 0; i < model->stringInputs; i++)
        feedValues[model->numericInputs + i] = stringTensor(strIns[i]);

    TF_SessionRun(model->session,
                  NULL,
                  model->feeds,
                  feedValues,
                  model->totalInputs,
                  model->fetches,
                  fetchValues,
                  1,
                  NULL,
                  0,
                  NULL,
                  status);

    Outputs* output = parseOutputs(fetchValues[0], status);

    TF_DeleteStatus(status);
    for (int i = 0; i < model->totalInputs; i++)
        TF_DeleteTensor(feedValues[i]);

    free(feedValues);
    free(fetchValues);

    return output;
}

void printOutput(Outputs* output) {
    printf("\n********** Result of Execution ************\n");

    if (output->statusCode != 0) {
        printf("Status code: %d\n", output->statusCode);
        printf("Message: %s", output->statusMessage);
    }
    else {
        if (output->nDims == 1) {
            printArray(output->data, 0, output->dims[0]);
        }
        else if (output->nDims == 2) {
            int start = 0;

            printf("[\n");
            for (int i = 0; i < output->dims[0]; i++) {
                printf("  ");
                printArray(output->data, start, output->dims[1]);
                printf("\n");
                start += output->dims[1];
            }
            printf("]");
        }
    }

    printf("\n");
}
