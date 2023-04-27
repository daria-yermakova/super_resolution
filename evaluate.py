
peek_index = 9

input_image = image_batch[peek_index]
target_image = target_batch[peek_index]

logits = model(input_image.unsqueeze(0).to(device))
reconstruction = torch.nn.functional.tanh(logits).cpu().detach().numpy().squeeze().transpose(1, 2, 0)

input_display = input_image.cpu().numpy().squeeze().transpose(1, 2, 0)
target_display = target_image.cpu().numpy().squeeze().transpose(1, 2, 0)

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].imshow(reconstruction[100:200, 100:200, :])
ax[0, 1].imshow(input_display[100:200, 100:200, :])
ax[0, 2].imshow(target_display[100:200, 100:200, :])

ax[1, 0].imshow(reconstruction[:, :, :])
ax[1, 1].imshow(input_display[:, :, :])
ax[1, 2].imshow(target_display[:, :, :])
plt.savefig(save_dir / "reconstruction_sample.jpg", dpi=300)