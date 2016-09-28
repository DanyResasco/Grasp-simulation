#include <apps/tridecimator/mvbb_decimator.h>

Eigen::MatrixXd MVBBDecimator(std::string filename)
{
    MyMesh mesh;

    int err=vcg::tri::io::Importer<MyMesh>::Open(mesh, filename.c_str());
    if(err)
    {
      exit(-1);
    }

    int FinalSize = mesh.fn / 20;

    MeshReductor meshreductor(mesh);
    meshreductor.reduceMesh(FinalSize);

    return meshreductor.getEigenMesh();
}
