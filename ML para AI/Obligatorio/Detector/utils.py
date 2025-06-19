from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import feature

def non_max_suppression(indices, sizes, overlapThresh, scores=None, use_scores=False):
    if len(indices) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    if indices.dtype.kind == "i":
        indices = indices.astype("float")

    pick = []

    x1 = np.array([indices[i, 0] for i in range(indices.shape[0])])
    y1 = np.array([indices[i, 1] for i in range(indices.shape[0])])
    Ni = np.array([sizes[i, 0] for i in range(sizes.shape[0])])
    Nj = np.array([sizes[i, 1] for i in range(sizes.shape[0])])
    x2 = x1 + Ni
    y2 = y1 + Nj
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if use_scores and scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return indices[pick].astype("int"), sizes[pick]

def sliding_window(img, patch_size=(64,64), istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Nj, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = resize(patch, patch_size)
            yield (i, j), patch

def detections_by_scale(test_image, test_scales, step, clf, scaler, pca, size=(64,64), thresholds=[0.5]):
    raw_detections = []
    detections = []

    for scale in tqdm(test_scales):
        raw_detections_scale = []
        detections_scale = []

        indices, patches = zip(*sliding_window(test_image, scale=scale, istep=step, jstep=step))
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        patches_hog = scaler.transform(patches_hog)
        patches_hog = pca.transform(patches_hog)
        indices = np.array(indices)

        for thr in thresholds:
            probas = clf.predict_proba(patches_hog)[:,1]
            labels = (probas >= thr).astype(int)
            raw_detections_scale.append(labels.sum())

            detecciones = indices[labels == 1]
            scores = probas[labels == 1]
            Ni, Nj = (int(scale * s) for s in size)
            sizes_array = np.array([(Ni, Nj)] * len(detecciones))

            detecciones, _ = non_max_suppression(
                detecciones,
                sizes_array,
                overlapThresh=0.3,
                scores=scores,
                use_scores=True
            )

            detections_scale.append(len(detecciones))

        raw_detections.append(raw_detections_scale)
        detections.append(detections_scale)

    return np.array(raw_detections), np.array(detections)

def evaluate_detections_by_scale(image, test_scales, clf, scaler, pca, thresholds=[0.5],
                                 patch_size=(64, 64), step=2, true_scale=None, number_faces=None):
    raw_detections, detections = detections_by_scale(
        image, test_scales, step, clf, scaler, pca,
        size=patch_size, thresholds=thresholds
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].set_title('Bruto')
    if true_scale is not None:
        ax[0].axvline(x=true_scale, ls='--', color='red')
    for i, thr in enumerate(thresholds):
        ax[0].step(test_scales, raw_detections[:, i], label=f'Thr={thr}')
    ax[0].grid(True)
    ax[0].set_xlabel('Escalas')
    ax[0].set_ylabel('Detecciones')
    ax[0].legend()

    ax[1].set_title('Procesado')
    if true_scale is not None:
        ax[1].axvline(x=true_scale, ls='--', color='red')
    if number_faces is not None:
        ax[1].axhline(y=number_faces, ls='--', color='red')
    for i, thr in enumerate(thresholds):
        ax[1].step(test_scales, detections[:, i], label=f'Thr={thr}')
    ax[1].grid(True)
    ax[1].set_xlabel('Escalas')
    ax[1].set_ylabel('Detecciones')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    return raw_detections, detections

def global_multiscale_detection(image, clf, scaler, pca, test_scales,
                                patch_size=(64, 64), threshold=0.1, step=2,
                                overlapThresh=0.3, plot=True):
    global_indices = []
    global_scores = []
    global_sizes = []

    for scale in tqdm(test_scales, desc="Escalas"):
        Ni, Nj = (int(scale * patch_size[0]), int(scale * patch_size[1]))
        indices, patches = zip(*sliding_window(image, scale=scale, istep=step, jstep=step))
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        patches_hog = scaler.transform(patches_hog)
        patches_hog = pca.transform(patches_hog)
        probas = clf.predict_proba(patches_hog)[:, 1]

        labels = (probas >= threshold).astype(int)
        selected_indices = np.array(indices)[labels == 1]
        selected_scores = probas[labels == 1]

        global_indices.extend(selected_indices)
        global_scores.extend(selected_scores)
        global_sizes.extend([(Ni, Nj)] * len(selected_indices))

    global_indices = np.array(global_indices)
    global_scores = np.array(global_scores)
    global_sizes = np.array(global_sizes)

    filtered_indices, filtered_sizes = non_max_suppression(
        global_indices,
        global_sizes,
        overlapThresh=overlapThresh,
        scores=global_scores,
        use_scores=True
    )

    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        for (i, j), (Ni, Nj) in zip(filtered_indices, filtered_sizes):
            ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='blue', alpha=0.6, lw=1, facecolor='none'))
        ax.set_title('Detecciones globales con NMS')
        plt.tight_layout()
        plt.show()

    return filtered_indices, filtered_sizes

def evaluate_detections_at_scale(image, clf, scaler, pca, scale,
                                 thresholds=[0.5, 0.1, 0.8],
                                 patch_size=(64, 64), overlapThresh=0.1, plot=True):
    indices, patches = zip(*sliding_window(image, scale=scale))
    patches_hog = np.array([feature.hog(patch) for patch in patches])
    patches_hog = scaler.transform(patches_hog)
    patches_hog = pca.transform(patches_hog)
    indices = np.array(indices)

    Ni, Nj = (int(scale * patch_size[0]), int(scale * patch_size[1]))
    size_array = np.array([(Ni, Nj)] * len(indices))

    detecciones_dict = {}

    for thr in thresholds:
        probas = clf.predict_proba(patches_hog)[:, 1]
        labels = (probas >= thr).astype(int)

        mask = labels == 1
        selected_indices = indices[mask]
        selected_sizes = size_array[mask]

        detecciones, _ = non_max_suppression(
            selected_indices,
            selected_sizes,
            overlapThresh=overlapThresh,
            scores=probas[mask],
            use_scores=True
        )

        detecciones_dict[thr] = detecciones

    if plot:
        fig, ax = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4))
        if len(thresholds) == 1:
            ax = [ax]
        for i, thr in enumerate(thresholds):
            ax[i].imshow(image, cmap='gray')
            ax[i].axis('off')
            for (y, x) in detecciones_dict[thr]:
                ax[i].add_patch(
                    plt.Rectangle((x, y), Nj, Ni, edgecolor='red', alpha=1, lw=1, facecolor='none')
                )
            ax[i].set_title(f'Thr={thr}')
        plt.tight_layout()
        plt.show()

    return detecciones_dict
