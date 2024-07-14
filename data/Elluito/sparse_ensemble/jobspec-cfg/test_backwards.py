# -*- coding: utf-8 -*-
# taken from
import torch
import math

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"


# torch.set_default_device(device)
def run():
    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype, requires_grad=False, device=device)
    y = torch.sin(x)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    a = torch.randn((), dtype=dtype, requires_grad=True, device=device)
    b = torch.randn((), dtype=dtype, requires_grad=True, device=device)
    c = torch.randn((), dtype=dtype, requires_grad=True, device=device)
    d = torch.randn((), dtype=dtype, requires_grad=True, device=device)
    # a.zero_grad()
    # b.zero_grad()
    # c.zero_grad()
    # d.zero_grad()

    learning_rate = 1e-6
    # x = x.cuda()
    # y = y.cuda()
    for t in range(2000):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        print("About to compute the backwards")
        loss.retain_grad()
        loss.backward()
        print("After computing the backwards")

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


def run2():
    # -*- coding: utf-8 -*-
    import torch
    import math

    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # Prepare the input tensor (x, x^2, x^3).
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use RMSprop; the optim package contains many other
    # optimization algorithms. The first argument to the RMSprop constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for t in range(2000):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(xx)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    linear_layer = model[0]
    print(
        f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


if __name__ == '__main__':
    run()
    run2()
