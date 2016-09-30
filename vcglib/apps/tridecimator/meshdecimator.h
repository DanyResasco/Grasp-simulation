// stuff to define the mesh


#include <vcg/complex/complex.h>

// io
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_ply.h>

// local optimization
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

using namespace vcg;
using namespace tri;

class MyVertex;
class MyEdge;
class MyFace;

struct MyUsedTypes: public UsedTypes<Use<MyVertex>::AsVertexType,Use<MyEdge>::AsEdgeType,Use<MyFace>::AsFaceType>{};

class MyVertex  : public Vertex< MyUsedTypes,
    vertex::VFAdj,
    vertex::Coord3f,
    vertex::Normal3f,
    vertex::Mark,
    vertex::Qualityf,
    vertex::BitFlags  >{
public:
  vcg::math::Quadric<double> &Qd() {return q;}
private:
  math::Quadric<double> q;
  };

class MyEdge : public Edge< MyUsedTypes> {};

typedef BasicVertexPair<MyVertex> VertexPair;

class MyFace    : public Face< MyUsedTypes,
  face::VFAdj,
  face::VertexRef,
  face::BitFlags > {};

// the main mesh class
class MyMesh    : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace> > {};


class MyTriEdgeCollapse: public vcg::tri::TriEdgeCollapseQuadric< MyMesh, VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>  > {
            public:
            typedef  vcg::tri::TriEdgeCollapseQuadric< MyMesh,  VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>  > TECQ;
            typedef  MyMesh::VertexType::EdgeType EdgeType;
            inline MyTriEdgeCollapse(  const VertexPair &p, int i, BaseParameterClass *pp) :TECQ(p,i,pp){}
};

class MeshReductor{
  MyMesh &mesh_;
public:
  MeshReductor(MyMesh &mesh):mesh_(mesh){};
  int reduceMesh(int FinalNoFaces);
  Eigen::MatrixXd getEigenVertices();
  Eigen::MatrixXi getEigenFaces();
};

int MeshReductor::reduceMesh(int FinalNoFaces)
{

  TriEdgeCollapseQuadricParameter qparams;
  qparams.QualityThr  =.3;
  float TargetError=std::numeric_limits<float>::max();

  int dup = tri::Clean<MyMesh>::RemoveDuplicateVertex(mesh_);
  int unref =  tri::Clean<MyMesh>::RemoveUnreferencedVertex(mesh_);

  vcg::tri::UpdateBounding<MyMesh>::Box(mesh_);

  // decimator initialization
  vcg::LocalOptimization<MyMesh> DeciSession(mesh_,&qparams);


  DeciSession.Init<MyTriEdgeCollapse>();


  DeciSession.SetTargetSimplices(FinalNoFaces);
  DeciSession.SetTimeBudget(0.5f);
  if(TargetError< std::numeric_limits<float>::max() ) DeciSession.SetTargetMetric(TargetError);

  while(DeciSession.DoOptimization() && mesh_.FN() > FinalNoFaces && DeciSession.currMetric < TargetError)
  {
    continue;
  }
  
  tri::Clean<MyMesh>::RemoveUnreferencedVertex(mesh_);
  vcg::tri::Allocator<MyMesh>::CompactFaceVector(mesh_);
  vcg::tri::Allocator<MyMesh>::CompactVertexVector(mesh_);
  
  return 1;
}

Eigen::MatrixXd MeshReductor::getEigenVertices()
{

  int n_vertex= mesh_.VN();
  Eigen::MatrixXd eigen_vertices(n_vertex, 3);
  
   for (int i = 0; i < n_vertex; i++)
   {
     eigen_vertices.block<1,3>(i,0) << mesh_.vert[i].cP()[0] , mesh_.vert[i].cP()[1] , mesh_.vert[i].cP()[2];
   }
  
  return eigen_vertices;
}

Eigen::MatrixXi MeshReductor::getEigenFaces()
{

  int n_face= mesh_.FN();
  int n_vertex= mesh_.VN();

  Eigen::MatrixXi eigen_faces(n_face, 3);
  
  unsigned int i_f = 0;
  for(MyMesh::FaceIterator fi = mesh_.face.begin(); fi!=mesh_.face.end(); ++fi )
  {
    if(fi->IsD()) 
    {
        std::cout << "!" << std::endl;
        continue;
    }
    
    int p_i[3];
    for(unsigned int i_v = 0; i_v < 3; ++i_v)
    {
        p_i[i_v]  = -1;

        for(unsigned int j_v = 0; j_v < n_vertex; ++j_v)
        {
            if(&(mesh_.vert[j_v]) == fi->cV(i_v))
            {
                p_i[i_v] = j_v;
            }
        }

        if(p_i[i_v] < 0) // brute force
        {
            for(unsigned int j_v = 0; j_v < n_vertex; ++j_v)
            {
                    if(mesh_.vert[j_v].P() == fi->cV(i_v)->P())
                    {
                        p_i[i_v] = j_v;
                        std::cout << "!";
                    }
            }
        }
            
        if(p_i[i_v] < 0)
        {
            std::cerr << "Error Finding Vertex " << i_v << " of face " << i_f << " with coords: " << fi->cV(i_v) << std::endl;
            std::cerr << "coords:" << fi->cV(i_v)->P()[0] << " "
                                   << fi->cV(i_v)->P()[1] << " "
                                   << fi->cV(i_v)->P()[2] << std::endl;
            exit(-1);
        }
    }
    eigen_faces.block<1,3>(i_f,0) << p_i[0] , p_i[1] , p_i[2];
        
    ++i_f;
  }
  return eigen_faces;
}
