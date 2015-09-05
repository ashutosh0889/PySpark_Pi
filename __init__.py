from flask import Flask, render_template
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
os.environ['SPARK_HOME'] = "/usr/local/spark-1.4.1-bin-hadoop2.6"
sys.path.append("/usr/local/spark-1.4.1-bin-hadoop2.6/python/")
# Now we are ready to import Spark Modules
try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print "Successfully Imported...!"
except ImportError as e:
    print "Error importing Spark Modules", e
    sys.exit(1)

sc = SparkContext()
# Getting the data from csv
csv = sc.textFile("multicollinearity.csv")
csv = csv.map(lambda l: l.split(","))
header = csv.first()
csvNH = csv.filter(lambda l: l != header)
"""csvNH.first()
t = csvNH.map(lambda l: float(l[1]))
a = t.collect()
t = csvNH.map(lambda l: float(l[2]))
b = t.collect()
t = csvNH.map(lambda l: float(l[3]))
c = t.collect()
t = csvNH.map(lambda l: float(l[4]))
d = t.collect()
t = csvNH.map(lambda l: float(l[5]))
e = t.collect()
t = csvNH.map(lambda l: float(l[0]))
s = t.collect()
header = header[1:]
x = np.concatenate(([a], [b]), axis=0)
x = np.concatenate((x, [c]), axis=0)
x = np.concatenate((x, [d]), axis=0)
x = np.concatenate((x, [e]), axis=0)
"""

data = csvNH.map(lambda p: (float(p[1]),float(p[2]),float(p[3]),float(p[4]),float(p[0])))
x = data.take(10)
print 'data: ',x
sum = []
min = []
max = []
mn = []
sd = []
i = 0
while(i < len(x)):
    sum = np.append(sum, [np.sum(x[i])])
    min = np.append(min, [np.min(x[i])])
    max = np.append(max, [np.max(x[i])])
    mn = np.append(mn, [np.mean(x[i])])
    sd = np.append(sd, [np.std(x[i])])
    i += 1

# Normalization by Standard Deviation
i = 0
nm = []
while(i < len(x)):
    mean = np.mean(x[i])
    std = np.std(x[i])
    j = 0
    temp = []
    while(j < len(x[i])):
        temp = np.append(temp, [(x[i][j]-mean)/std])
        j += 1
    nm = np.append(nm, [temp])
    i += 1
nm = np.reshape(nm, (5, 373))
print 'Normalize Data:', nm
corr = np.corrcoef(nm)
# print corr
pca = decomposition.PCA(copy=True, n_components=5, whiten=False)
pca.fit(nm.T)
p = pca.explained_variance_ratio_
print 'over all pca:', p
# pf=map("{0:.32f}".format, p)
# print pf
z = np.arange(1, len(p)+1, 1)
# plt.plot(z,p)
# plt.title('over all pca')
# plt.show()

eigenvector = pca.components_
print 'eigenvector', eigenvector

eigenvalue = pca.explained_variance_
print 'eigenvalue', eigenvalue
eigenvaluesqrt = np.sqrt(eigenvalue)
print 'sqrt of eigen values', eigenvaluesqrt
i = 0
loadingfactor = []
while(i < len(eigenvaluesqrt)):
    j = 0
    while(j < len(eigenvector[i])):
        loadingfactor = np.append(loadingfactor, [eigenvaluesqrt[i]*eigenvector[i][j]])
        j += 1
    i += 1

loadingfactor = np.reshape(loadingfactor, (5, 5))
print 'LoadingFactor', loadingfactor

mx = np.matrix(eigenvector)
my = np.matrix(nm.T)
# print mx,my
pc = my*mx
print 'pca obsevations', pc
# ac=pc * (mx.T)
# print ac
pc1 = np.array(pc.T)
i = 0
loadingOnObsevation = []
while(i < len(eigenvaluesqrt)):
    j = 0
    while(j < len(pc1[i])):
        loadingOnObsevation = np.append(loadingOnObsevation, [eigenvaluesqrt[i]*pc1[i][j]])
        j += 1
    i += 1

loadingOnObsevation = np.reshape(loadingOnObsevation, (5, 373))
# print 'loadings on pc observation', loadingOnObsevation

# Communality on loading factors

l = loadingfactor
i = 0
cm = []
while(i < len(loadingfactor)):
    j = 0
    add = 0
    while(j < len(loadingfactor[i])):
        add = add+loadingfactor[i][j]*loadingfactor[i][j]
        j += 1
    cm = np.append(cm, [add])
    i += 1

print 'Communalities', cm

# rotation varimax of loadings


def varimax(Phi, gamma=1, q=20, tol=1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d/d_old < tol:
            break
    return dot(Phi, R)
varimax = varimax(loadingfactor.T)
print 'varimax', varimax

# Calculating regression on pca using y as response


def desc_graph(a):
    i = 0
    while (i < (len(a)-1)):
        if a[i] == a[i+1]:
            del a[i]
        else:
            i = i+1
    gradients = np.diff(a)
    # print gradients
    max_num = 0
    min_num = 0
    max_loc = []
    min_loc = []
    count = 0
    for i in gradients[:-1]:
        count += 1
        if((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) & (i != gradients[count])):
            max_num += 1
            max_loc.append(count)
        if((cmp(i, 0) < 0) & (cmp(gradients[count], 0) > 0) & (i != gradients[count])):
            min_num += 1
            min_loc.append(count)

    if((max_loc[len(max_loc)-1] <= (len(a)-1)) | (min_loc[len(min_loc)-1] <= (len(a)-1))):
        if(a[max_loc[len(max_loc)-1]] < a[len(a)-1]):
            max_num += 1
            max_loc.append(len(a)-1)
        else:
            min_num += 1
            min_loc.append(len(a)-1)

    peak = []
    dip = []
    j = 0
    while(j < len(max_loc)):
        peak = np.append(peak, a[max_loc[j]])
        j += 1

    k = 0
    while(k < len(min_loc)):
        dip = np.append(dip, a[min_loc[k]])
        k += 1

    if(min_loc[0] > max_loc[0]):
        v = a[0]
        dip = np.concatenate(([v], dip), axis=0)
        min_num += 1
        min_loc = np.concatenate(([0], min_loc), axis=0)
    else:
        v = a[0]
        peak = np.concatenate(([v], peak), axis=0)
        max_num += 1
        max_loc = np.concatenate(([0], max_loc), axis=0)

    if(max_num > min_num):
        peak = np.delete(peak, 0)
        max_num -= 1
        max_loc = np.delete(max_loc, 0)
    else:
        dip = np.delete(dip, 0)
        min_num -= 1
        i += 1
        min_loc = np.delete(min_loc, 0)

    print 'No. of Peaks', max_num, 'No. of Dips', min_num
    return peak, dip, max_loc, min_loc

peak, dip, max_loc, min_loc = desc_graph(s)
# print peak
# print dip

# pc for peak
i = 0
peak_pc = []
while(i < len(max_loc)):
    peak_pc = np.append(peak_pc, [pc[max_loc[i]]])
    i += 1
peak_pc = np.reshape(peak_pc, (117, 5))
# print 'peak_pc',peak_pc

# pc for dip

i = 0
dip_pc = []
while(i < len(min_loc)):
    dip_pc = np.append(dip_pc, [pc[min_loc[i]]])
    i += 1
dip_pc = np.reshape(dip_pc, (117, 5))
# print 'dip_pc',dip_pc

znum = np.arange(1, len(peak_pc)+1, 1)
plt.plot(znum, peak_pc.T[0], znum, dip_pc.T[0])
plt.title('PCA1')
plt.savefig('static/images/pca1.png', format='png')
plt.show()
plt.plot(znum, peak_pc.T[1], znum, dip_pc.T[1])
plt.title('PCA2')
plt.savefig('static/images/pca2.png', format='png')
plt.show()
plt.plot(znum, peak_pc.T[2], znum, dip_pc.T[2])
plt.title('PCA3')
plt.savefig('static/images/pca3.png', format='png')
plt.show()
plt.plot(znum, peak_pc.T[3], znum, dip_pc.T[3])
plt.title('PCA4')
plt.savefig('static/images/pca4.png', format='png')
plt.show()
plt.plot(znum, peak_pc.T[4], znum, dip_pc.T[4])
plt.title('PCA5')
plt.savefig('static/images/pca5.png', format='png')
plt.show()

i = 0
peak_x = []
while(i < len(max_loc)):
    peak_x = np.append(peak_x, [x.T[max_loc[i]]])
    i += 1

peak_x = np.reshape(peak_x, (117, 5))
# print 'peak_x',peak_x

i = 0
dip_x = []
while(i < len(min_loc)):
    dip_x = np.append(dip_x, [x.T[min_loc[i]]])
    i += 1

dip_x = np.reshape(dip_x, (117, 5))
# print 'dip_x',dip_x

plt.plot(znum, peak_x.T[0], znum, dip_x.T[0])
plt.title('x1_peak_dip')
plt.savefig('static/images/x1_peak_dip.png', format='png')
plt.show()
plt.plot(znum, peak_x.T[1], znum, dip_x.T[1])
plt.title('x2_peak_dip')
plt.savefig('static/images/x2_peak_dip.png', format='png')
plt.show()
plt.plot(znum, peak_x.T[2], znum, dip_x.T[2])
plt.title('x3_peak_dip')
plt.savefig('static/images/x3_peak_dip.png', format='png')
plt.show()
plt.plot(znum, peak_x.T[3], znum, dip_x.T[3])
plt.title('x4_peak_dip')
plt.savefig('static/images/x4_peak_dip.png', format='png')
plt.show()
plt.plot(znum, peak_x.T[4], znum, dip_x.T[4])
plt.title('x5_peak_dip')
plt.savefig('static/images/x5_peak_dip.png', format='png')
plt.show()


regraw = np.linalg.lstsq(nm.T, s)[0]
print 'reg on row data', regraw
regression = np.linalg.lstsq(pc, s)[0]
print 'reg on pca observation', regression
regpeak = np.linalg.lstsq(peak_pc, peak)[0]
print 'reg-peak', regpeak
regdip = np.linalg.lstsq(dip_pc, dip)[0]
print 'reg-dip', regdip

# WebApp
app = Flask(__name__)


@app.route('/')
def homepage():

    title = "Principle Component Analysis"
    paragraph = ["wow I am learning so much great stuff!", "wow I am learning so much great stuff!"]

    try:
        return render_template("index.html", title=title, paragraph=paragraph, header=header, x=x, sum=sum, max=max,
                               min=min, mean1=mn, std1=sd, nm=nm, corr=corr, pca=p, eigvec=eigenvector,
                               eigval=eigenvalue, loadingfactor=loadingfactor, pcaobservation=pc1, communalities=cm,
                               varimax=varimax, rw=regraw, rp=regression, rpkpc=regpeak, rdppc=regdip)
    except Exception, e:
        return str(e)


@app.route('/about')
def aboutpage():

    title = "About this site"
    paragraph = ["Progen Business Solution!!!"]

    pagetype = 'about'

    return render_template("index.html", title=title, paragraph=paragraph, pageType=pagetype)


@app.route('/about/contact')
def contactpage():

    title = "About this site"
    paragraph = ["mohit.gupta@progenbusiness.com"]

    pageType = 'about'

    return render_template("index.html", title=title, paragraph=paragraph, pageType=pageType)


if __name__ == "__main__":
    app.run()
