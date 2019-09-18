import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names, ylen, xlen, savename=""):

    plt.figure(figsize=(2+(5*xlen), 5*ylen))

    for y in range(ylen):
        for x in range(xlen):
            plt.subplot(2,3,(y*3)+(x+1))
            plt.title(names[(y*3)+x])
            plt.imshow(dat2canvas(data=contents[(y*3)+x]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="results")
    result_list = ["tr_resotring", "tr_latent_z", "tr_latent_z_T"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        x_restore, x_Trestore = sess.run([neuralnet.x_r, neuralnet.x_Tr], \
            feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
        z_enc, z_T_enc = sess.run([neuralnet.z_pack[0], neuralnet.z_T_pack[0]], \
            feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})

        if(neuralnet.z_dim == 2):
            latent_plot(latent=z_enc, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent_z", "%08d.png" %(epoch)))
            latent_plot(latent=z_T_enc, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent_z_T", "%08d.png" %(epoch)))
        else:
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(z_enc)
            latent_plot(latent=pca_features, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent_z", "%08d.png" %(epoch)))
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(z_T_enc)
            latent_plot(latent=pca_features, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent_z_T", "%08d.png" %(epoch)))

        save_img(contents=[x_tr, x_restore, (x_tr-x_restore)**2, \
            x_restore, x_Trestore, (x_restore-x_Trestore)**2], \
            names=["x", "x_r", "(x-x_r)^2", \
            "x_r", "x_Tr", "(x_r-x_Tr)^2"], \
            ylen=2, xlen=3, \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            _ = sess.run(neuralnet.optimizer1, \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            _ = sess.run(neuralnet.optimizer2, \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            summaries = sess.run(neuralnet.summaries, \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]}, \
                options=run_options, run_metadata=run_metadata)

            loss_T, loss_G, loss_E, loss_tot = sess.run([neuralnet.loss_T_mean, neuralnet.loss_G_mean, neuralnet.loss_E_mean, neuralnet.loss_tot], \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})

            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration) Loss  T:%.3f, G:%.3f, E:%.3f, Tot:%.3f" \
            %(epoch, epochs, iteration, loss_T, loss_G, loss_E, loss_tot))

        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        print("\nRestoring parameters")
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    scores_normal, scores_abnormal = [], []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.

        x_restore, score_anomaly = sess.run([neuralnet.x_r, neuralnet.mse_r], \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})
        if(y_te[0] == 1): scores_normal.append(score_anomaly[0])
        else: scores_abnormal.append(score_anomaly[0])

        if(terminator): break

    scores_normal = np.asarray(scores_normal)
    scores_abnormal = np.asarray(scores_abnormal)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    abnormal_avg, abnormal_std = np.average(scores_abnormal), np.std(scores_abnormal)
    print("Noraml  avg: %.5f, std: %.5f" %(normal_avg, normal_std))
    print("Abnoraml  avg: %.5f, std: %.5f" %(abnormal_avg, abnormal_std))
    outbound = normal_avg + (normal_std * 3)
    print("Outlier boundary of normal data: %.5f" %(outbound))

    plt.hist(scores_normal, alpha=0.5, label='Normal')
    plt.hist(scores_abnormal, alpha=0.5, label='Abnormal')
    plt.legend(loc='upper right')
    plt.savefig("histogram-test.png")
    plt.close()

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_enc_tot, z_T_enc_tot, y_te_tot = None, None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.

        x_restore, score_anomaly = sess.run([neuralnet.x_r, neuralnet.mse_r], \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})
        z_enc, z_T_enc = sess.run([neuralnet.z_pack[0], neuralnet.z_T_pack[0]], \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})

        loss4box[y_te[0]].append(score_anomaly)

        if(z_enc_tot is None):
            z_enc_tot = z_enc
            z_T_enc_tot = z_T_enc
            y_te_tot = y_te
        else:
            z_enc_tot = np.append(z_enc_tot, z_enc, axis=0)
            z_T_enc_tot = np.append(z_T_enc_tot, z_T_enc, axis=0)
            y_te_tot = np.append(y_te_tot, y_te, axis=0)

        outcheck = score_anomaly > outbound
        fcsv.write("%d, %.5f, %r\n" %(y_te, score_anomaly, outcheck))

        [h, w, c] = x_restore[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_restore[0]
        canvas[:, w*2:, :] = (x_te[0]-x_restore[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    boxplot(contents=loss4box, savename="test-box.png")

    if(neuralnet.z_dim == 2):
        latent_plot(latent=z_enc_tot, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent_z.png"))
        latent_plot(latent=z_T_enc_tot, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent_z_T.png"))
    else:
        pca = PCA(n_components=2)
        tot_vec = np.append(z_enc_tot, z_T_enc_tot, axis=0)
        pca_features = pca.fit_transform(tot_vec)
        latent_plot(latent=pca_features[:int(pca_features.shape[0]/2)], y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent_z.png"))
        latent_plot(latent=pca_features[int(pca_features.shape[0]/2):], y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent_z_T.png"))
