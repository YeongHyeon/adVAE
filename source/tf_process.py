import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

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
    bp = ax1.boxplot(data, showfliers=True)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="results")
    result_list = ["tr_resotring"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        x_restore = sess.run(neuralnet.x_r, \
            feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})

        save_img(contents=[x_tr, x_restore, (x_tr-x_restore)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            loss_T, loss_G, loss_E, loss_tot = sess.run([neuralnet.loss_T_mean, neuralnet.loss_G_mean, neuralnet.loss_E_mean, neuralnet.loss_tot], \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            print("Epoch [%d / %d] (%d iteration) Loss  T:%.3f, G:%.3f, E:%.3f, Tot:%.3f" \
                %(epoch, epochs, iteration, loss_T, loss_G, loss_E, loss_tot))

            _ = sess.run(neuralnet.optimizer1, \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            _ = sess.run(neuralnet.optimizer2, \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            summaries = sess.run(neuralnet.summaries, \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]}, \
                options=run_options, run_metadata=run_metadata)

            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break


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

    loss_list = []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.

        x_restore, score_anomaly = sess.run([neuralnet.x_r, neuralnet.mse_r], \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})
        if(y_te[0] == 1):
            loss_list.append(score_anomaly[0])

        if(terminator): break

    loss_list = np.asarray(loss_list)
    loss_avg, loss_std = np.average(loss_list), np.std(loss_list)
    outbound = loss_avg + (loss_std * 3)
    print("Loss  avg: %.5f, std: %.5f" %(loss_avg, loss_std))
    print("Outlier boundary: %.5f" %(outbound))

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_enc_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.

        x_restore, score_anomaly = sess.run([neuralnet.x_r, neuralnet.mse_r], \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})

        loss4box[y_te[0]].append(score_anomaly)

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
