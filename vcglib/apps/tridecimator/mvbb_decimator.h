#include <apps/tridecimator/meshdecimator.h>
#include <string>

class MVBBDecimator
{
public:
    MVBBDecimator();
    ~MVBBDecimator();

    void decimateTriMesh(std::string filename, 
                         int targetFacesNumber = 1000);
    void decimateTriMesh(const Eigen::MatrixXd& vertices,
                         const Eigen::MatrixXi& faces,
                         int targetFacesNumber = 1000);

    /**
     * @brief MeshReductor::getEigenVertices returns decimated vertices
     * @return (n_verticesx3 matrix)
     */
    Eigen::MatrixXd getEigenVertices();

    /**
     * @brief MeshReductor::getEigenFaces returns decimated faces
     * @return (n_facesx3 matrix)
     */
    Eigen::MatrixXi getEigenFaces();

private:
    void decimate(int FinalSize);

    MeshReductor* meshreductor;
    MyMesh mesh;
};

