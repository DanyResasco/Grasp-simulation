#include <apps/tridecimator/mvbb_decimator.h>
#include <algorithm>

MVBBDecimator::MVBBDecimator() : meshreductor(NULL) {}

MVBBDecimator::~MVBBDecimator()
{
    if(meshreductor != NULL)
        delete meshreductor;
}

void MVBBDecimator::decimateTriMesh(string filename)
{
    int err=vcg::tri::io::Importer<MyMesh>::Open(mesh, filename.c_str());
    if(err)
    {
      exit(-1);
    }

    this->decimate();
}

void MVBBDecimator::decimateTriMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &faces)
{
    MyMesh::VertexIterator vi = vcg::tri::Allocator<MyMesh>::AddVertices(mesh, vertices.rows());
    MyMesh::FaceIterator fi = vcg::tri::Allocator<MyMesh>::AddFaces(mesh, faces.rows());

    std::vector<MyMesh::VertexPointer> ivp(vertices.rows(), NULL);
    for(unsigned int i = 0; i < vertices.rows(); ++i)
    {
        ivp[i]=&*vi; vi->P()=MyMesh::CoordType ( vertices(i, 0),
                                                 vertices(i, 1),
                                                 vertices(i, 2));
        ++vi;
    }

    for(unsigned int i = 0; i < faces.rows(); ++i)
    {
        fi->V(0)=ivp[faces(i, 0)];
        fi->V(1)=ivp[faces(i, 1)];
        fi->V(2)=ivp[faces(i, 2)];
        ++fi;
    }

    this->decimate();
}

Eigen::MatrixXd MVBBDecimator::getEigenVertices()
{
    return meshreductor->getEigenVertices();
}

Eigen::MatrixXi MVBBDecimator::getEigenFaces()
{
    return meshreductor->getEigenFaces();
}

void MVBBDecimator::decimate()
{
    //int FinalSize = mesh.FN() / 4;
    int FinalSize = 1000;
    meshreductor = new MeshReductor(mesh);
    meshreductor->reduceMesh(FinalSize);
}
