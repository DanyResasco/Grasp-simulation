#include <apps/tridecimator/meshdecimator.h>

int main(int argc ,char**argv)
{

  MyMesh mesh;
  
  int FinalSize=1000;
  int err=vcg::tri::io::Importer<MyMesh>::Open(mesh,argv[1]);
  if(err)
  {
    exit(-1);
  }

  MeshReductor meshreductor(mesh);
  meshreductor.reduceMesh(FinalSize);

//   Eigen::MatrixXd eigen_points = meshreductor.getEigenMesh();
  vcg::tri::io::ExporterPLY<MyMesh>::Save(mesh,"reduced.ply");
  
    return 0;

}
