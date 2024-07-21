#ifndef NN_H
#define NN_H

#include <stdio.h>

typedef enum
{
    NO_OP,
    ADD,
    MUL,
    TANH,
} Operation;

typedef struct Value_
{
    double data;
    double grad;
    Operation operation;
    struct Value_ *first;
    struct Value_ *second;
} Value;

typedef struct
{
    size_t num_inputs;
    Value **weights;
    Value *bias;
} Node;

typedef struct
{
    size_t num_inputs;
    size_t num_nodes;
    Node **nodes;
} Layer;

typedef struct
{
    size_t num_layers;
    Layer **layers;
} NN;

Value* value_create(double data);
void value_free(Value *v);
void value_free_recursive(Value *v);
Value* value_operation(Value *first, Value *second, Operation op);
void value_init_backprop(Value *v);
void value_init_nudge(Value *v, double step_size);

Node* node_create(size_t num_inputs);
void node_free(Node *n);
Value* node_forward(Node *n, Value **inputs);

Layer* layer_create(size_t num_inputs, size_t num_outputs);
void layer_free(Layer *l);
Value** layer_forward(Layer *l, Value **inputs);

NN* nn_create(size_t *arch, size_t num_layers);
void nn_free(NN *nn);
Value** nn_forward(NN *nn, Value **inputs);

#endif