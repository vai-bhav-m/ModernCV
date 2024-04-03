import matplotlib.pyplot as plt
import pickle

file_path = "model_perf4layer.pkl"
with open(file_path, 'rb') as file:
    loaded_list = pickle.load(file)

losses, train_accs, test_accs = loaded_list

fig, ax = plt.subplots(1, 3, figsize=(18,5))
ax[0].plot(losses)
ax[0].set_ylabel("Loss")
ax[0].set_title("Neg. Log Likelihood Loss growth during training")

ax[1].plot(test_accs[0], label="Validation Accuracy")
ax[1].plot(train_accs, label="Train Accuracy")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Word-to-Word Transliteration Accuracy of the Model")
ax[1].legend() 

ax[2].plot(test_accs[1])
ax[2].set_ylabel("Levenshtein Accuracy")
ax[2].set_title("Levenshtein Distance based Accuracy on Validation Set")

plt.subplots_adjust(wspace=0.3)
# fig.suptitle()
fig.text(0.5, 0.02, "Performance of the RNN over training time", ha='center', fontsize=12)
plt.legend()
plt.show()
plt.savefig("plot4layerv2.png")
plt.close()