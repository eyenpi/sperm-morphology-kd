import torch


def fgsm_attack(model, vgg, loss, images, eps):
    images = images.reshape(-1, 1, 64, 64).cuda()

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    images.requires_grad = True

    outputs1 = model.forward(images)
    outputs2 = vgg(images)
    model.zero_grad()
    cost = loss(outputs1, outputs2).cuda()
    cost.backward()

    attacked_images = images + eps * images.grad.sign()
    return attacked_images
