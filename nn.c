#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn.h"

void _value_backprop(Value *v);
void _value_compute_gradients(Value *v);
void _value_init_grad_reset(Value *v);

// =================================================================
// MARK: Utils
// =================================================================

// Return a random double between -1.0 and 1.0
double getRandDouble(void)
{
    double val = (double)rand() / (double)RAND_MAX;
    return 2.0 * val - 1.0;
}

// =================================================================
// MARK: Value Impl
// =================================================================

Value* value_create(double data)
{
    Value *ret = malloc(sizeof(Value));
    assert(ret);

    ret->data = data;
    ret->grad = 0.0f;
    ret->operation = NO_OP;
    ret->first = NULL;
    ret->second = NULL;
    return ret;
}

void value_free(Value *v)
{
    if (!v) return;

    free(v);
    v->first = NULL;
    v->second = NULL;
    v = NULL;
}

Value* value_operation(Value *first, Value *second, Operation op)
{
    assert(first);

    Value *result = value_create(0.0);
    switch (op)
    {
    case ADD:
        result->data = first->data + second->data;
        break;
    case MUL:
        result->data = first->data * second->data;
        break;
    case TANH:
        assert(second == NULL);
        result->data = tanh(first->data);
        break;
    default:
        assert(0);
        break;
    }

    result->operation = op;
    result->first = first;
    result->second = second;
    return result;
}

void _value_init_grad_reset(Value *v)
{
    v->grad = 0.0;
    if (v->first) 
        _value_init_grad_reset(v->first);
    if (v->second) 
        _value_init_grad_reset(v->second);
}

void value_init_nudge(Value *v, double step_size)
{
    v->data -= step_size * v->grad;
    if (v->first) 
        value_init_nudge(v->first, step_size);
    if (v->second) 
        value_init_nudge(v->second, step_size);
}

// TODO: This calls backprop recursively, 
// It doesn't use topological sorting, 
// This might not work for DAGs
void value_init_backprop(Value *v)
{
    _value_init_grad_reset(v);

    v->grad = 1.0;
    _value_backprop(v);
}

void _value_backprop(Value *v)
{
    _value_compute_gradients(v);
    if (v->first) 
        _value_backprop(v->first);
    if (v->second) 
        _value_backprop(v->second);
}

void _value_compute_gradients(Value *v)
{
    // Leaf node
    if (v->first == NULL)
    {
        return;
    }

    switch (v->operation)
    {
    case ADD:
        v->first->grad += v->grad;
        v->second->grad += v->grad;
        break;
    case MUL:
        v->first->grad += (v->grad * v->second->data);
        v->second->grad += (v->grad * v->first->data);
        break;
    case TANH:
        v->first->grad += (1.0f - (v->data * v->data)) * v->grad;
        break;
    default:
        assert(0);
        break;
    }
}

// =================================================================
// MARK: Node Impl
// =================================================================

Node* node_create(size_t num_inputs)
{
    Node *n = malloc(sizeof(Node));
    n->bias = value_create(getRandDouble());
    n->weights = malloc(num_inputs * sizeof(Value*));
    n->num_inputs = num_inputs;

    for (int i = 0; i < num_inputs; i ++)
    {
        n->weights[i] = value_create(getRandDouble());
    }

    return n;
}

void node_free(Node *n)
{
    for (int i = 0; i < n->num_inputs; i ++)
    {
        value_free(n->weights[i]);
    }

    value_free(n->bias);
    free(n->weights);
    free(n);
    n = NULL;
}

Value* node_forward(Node *n, Value **inputs)
{
    Value *temp = n->bias;
    for (int i = 0; i < n->num_inputs; i++)
    {
        temp = value_operation(
            temp, 
            value_operation(inputs[i], n->weights[i], MUL),
            ADD
        );
    }

    return value_operation(temp, NULL, TANH);
}

// =================================================================
// MARK: Layer Impl
// =================================================================

Layer* layer_create(size_t num_inputs, size_t num_outputs)
{
    Layer *layer = malloc(sizeof(Layer));
    assert(layer);

    layer->num_inputs = num_inputs;
    layer->num_nodes = num_outputs;
    layer->nodes = malloc(num_outputs * sizeof(Node*));
    for (int i = 0;i < num_outputs; i ++)
    {
        layer->nodes[i] = node_create(num_inputs);
    }

    return layer;
}

void layer_free(Layer *l)
{
    for (int i = 0;i < l->num_nodes; i ++)
    {
        node_free(l->nodes[i]);
    }
    free(l->nodes);
    free(l);
    l = NULL;
}

Value** layer_forward(Layer *l, Value **inputs)
{
    Value **outputs = malloc(l->num_nodes * sizeof(Value*));
    for (int i = 0;i < l->num_nodes; i ++)
    {
        outputs[i] = node_forward(l->nodes[i], inputs);
    }

    return outputs;
}

// =================================================================
// MARK: NN Impl
// =================================================================

NN* nn_create(size_t *arch, size_t num_layers)
{
    NN *nn = malloc(sizeof(NN));
    nn->layers = malloc((num_layers - 1) * sizeof(Layer*));
    nn->num_layers = num_layers;

    for (int i = 0; i < num_layers - 1; i ++)
    {
        nn->layers[i] = layer_create(arch[i], arch[i + 1]);
    }

    return nn;
}

void nn_free(NN* nn)
{
    for (int i = 0; i < nn->num_layers - 1; i ++)
    {
        layer_free(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

Value** nn_forward(NN *nn, Value **inputs)
{
    Value **outputs = inputs;
    for (int i = 0; i < nn->num_layers - 1; i ++)
    {
        outputs = layer_forward(nn->layers[i], outputs);
    }
    return outputs;
}
