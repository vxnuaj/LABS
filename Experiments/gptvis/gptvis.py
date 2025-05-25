import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA

import matplotlib.gridspec as gridspec
import gudhi as gd

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# === CONFIG ===
prompt = "Explain the theory of relativity to me, from a quantum mechanical perspective."
num_generate = 200       # adjust if memory constrained
alpha_min = 0.05
device = "mps"  # change to "cuda" or "cpu" as needed

# === Load GPT-2
print('Loading Model')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
print('Model Loaded')

# === Generate + sync hidden states + text
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
generated_ids = input_ids.clone()
hidden_vectors = []
frame_texts = []

print('Generating text and collecting hidden states')

with torch.no_grad():
    for _ in range(num_generate):
        outputs = model(generated_ids, output_hidden_states=True)

        print(tokenizer.decode(generated_ids[0, -1].item()), flush = True, sep = '')

        # collect all 12 layer vectors for the current token
        for layer_hidden in outputs.hidden_states[1:]:
            last_vector = layer_hidden[0, -1, :]
            hidden_vectors.append(last_vector.cpu().numpy())

        # update the current text shown on screen
        frame_texts.append(tokenizer.decode(generated_ids[0]))

        # generate next token (sampling)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

print('Text generation complete')
print(f"Generated {len(frame_texts)} tokens")

# === Project to 3D
print('Projecting to 3D')
hidden_matrix = np.stack(hidden_vectors)
points_3d = PCA(n_components=3).fit_transform(hidden_matrix)

# === Compute persistence diagrams for each frame
print('Computing persistence diagrams')
pers_diagrams = []
for i in tqdm(range(num_generate)):
    end = (i + 1) * 12
    segment = points_3d[:end]
    rips_complex = gd.RipsComplex(points=segment, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    pers_diagrams.append(simplex_tree.persistence_intervals_in_dimension(1))  # H1: loops

# === Build animation frames
print('Building animation frames')
frames = []
cmap = colormaps["plasma"]
for i in range(num_generate):
    end = (i + 1) * 12
    segment = points_3d[:end]
    alphas = [alpha_min + (1 - alpha_min) * (j / end) for j in range(end - 1)]
    frames.append((segment, alphas, frame_texts[i], pers_diagrams[i]))

# === Plot & Animate
print('Plotting')
fig = plt.figure(figsize=(30, 20))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
ax3d = fig.add_subplot(gs[0], projection='3d')
axpd = fig.add_subplot(gs[1])
fig.subplots_adjust(top=0.80)
text_handle = ax3d.text2D(
    0.5, 0.60, "", transform=ax3d.transAxes,
    ha="center", va="top", fontsize=14, color='white', wrap=True
)

def update(frame_idx):
    ax3d.cla()
    ax3d.set_facecolor("#000000")
    ax3d.axis('off')
    axpd.cla()
    axpd.set_title("Persistence Diagram (H1)")
    axpd.set_xlabel("Birth")
    axpd.set_ylabel("Death")

    segment, alphas, current_text, pers_diag = frames[frame_idx]
    for i in range(len(segment) - 1):
        x = [segment[i, 0], segment[i + 1, 0]]
        y = [segment[i, 1], segment[i + 1, 1]]
        z = [segment[i, 2], segment[i + 1, 2]]
        ax3d.plot(x, y, z, color=cmap(i / len(points_3d)), linewidth=2, alpha=alphas[i])

    last_tokens = tokenizer.decode(tokenizer.encode(current_text)[-10:])
    text_handle.set_text(last_tokens)
    ax3d.add_artist(text_handle)

    # Plot persistence diagram for H1 (loops)
    if len(pers_diag) > 0:
        births = pers_diag[:, 0]
        deaths = pers_diag[:, 1]
        axpd.scatter(births, deaths, c='red')
        axpd.plot([0, np.max(deaths)], [0, np.max(deaths)], 'k--', alpha=0.5)
    axpd.set_xlim(0, np.max(points_3d))
    axpd.set_ylim(0, np.max(points_3d))
    return []

print('Animating')
ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=100, repeat=True)

# === Save or show
# ani.save("gpt2_synced_trail.mp4", writer="ffmpeg", dpi=200)

plt.show()  # Uncomment to preview live instead of savin