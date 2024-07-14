#pragma once

#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) 
#define APIEXPORT __declspec(dllexport)
#else
#define APIEXPORT
#endif


#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <chrono/core/ChMath.h>

#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>

struct xml_option {
    xml_option(std::string type_, std::string parameter_, std::string value_) : type(type_), parameter(parameter_), value(value_) {}
    std::string type;
    std::string parameter;
    std::string value;
};

class APIEXPORT MitsubaGenerator {
  public:
    MitsubaGenerator();
    MitsubaGenerator(const std::string& filename);
    ~MitsubaGenerator() {}
    void Open(const std::string& filename);
    void ExportDriver(const std::string& filename);
    // Function creates a node of a specific type
    void CreateNewNode(const std::string& type);
    void CloseNode();

    void AddAttribute(const std::string& name, const std::string& value);
    void AddAttributes(const std::vector<xml_option>& options);
    void CreatePlugin(const std::string& name, const std::string& type, const std::vector<xml_option>& options);
    void CreateTransform(const std::string& name);
    void CreateTransform(const chrono::ChVector<>& scale, const chrono::ChVector<>& position, const chrono::ChQuaternion<>& rotation);
    void AddIntegrator(std::string integrator = "path",
                       std::vector<xml_option> integrator_options = {xml_option("boolean", "hideEmitters", "true"), xml_option("integer", "maxDepth", "10"),
                                                                     xml_option("integer", "rrDepth", "10")});
    void AddInclude(std::string filename);

    void AddEmitter(std::string emitter = "sky",
                    std::vector<xml_option> emitter_options = {xml_option("float", "scale", "3"), xml_option("float", "hour", "12"),
                                                               xml_option("float", "turbidity", "10"), xml_option("spectrum", "albedo", ".15")},
                    const chrono::ChVector<>& scale = chrono::ChVector<>(1, 1, 1),
                    const chrono::ChVector<>& position = chrono::ChVector<>(0, 0, 0),
                    const chrono::ChQuaternion<>& rotation = chrono::ChQuaternion<>(1, 0, 0, 0));
    std::string CreateTriplet(const chrono::ChVector<>& v);
    void CreateXYZTriplet(const chrono::ChVector<>& v);
    void Translate(const chrono::ChVector<>& vector);
    void Scale(const chrono::ChVector<>& vector);
    void Rotate(const double angle, const chrono::ChVector<>& axis);
    void LookAt(const chrono::ChVector<>& origin, const chrono::ChVector<>& look_at, const chrono::ChVector<>& up);

    void CreateScene(bool add_integrator = true, bool add_sensor = true, bool add_sky = true);
    void Write();
    void SetDataFolder(const std::string& folder) { data_folder = folder; }
    void SetRenderFolder(const std::string& folder) { render_folder = folder; }
    void AddMesh(const std::string& filename,
                 const std::string& material,
                 const chrono::ChVector<>& scale,
                 const chrono::ChVector<>& position,
                 const chrono::ChQuaternion<>& rotation);
    void AddShape(const std::string& id, const chrono::ChVector<>& scale, const chrono::ChVector<>& position, const chrono::ChQuaternion<>& rotation);
    void AddCompleteShape(const std::string& id,
                          const std::string& material_type,
                          const chrono::ChVector<>& color,
                          const chrono::ChVector<>& scale,
                          const chrono::ChVector<>& position,
                          const chrono::ChQuaternion<>& rotation);
    void AddSimpleShape(const int type, const chrono::ChVector<>& scale, const chrono::ChVector<>& position, const chrono::ChQuaternion<>& rotation);

    void AddSensor(chrono::ChVector<> origin,
                   chrono::ChVector<> target,
                   chrono::ChVector<> up,
                   std::vector<std::tuple<int, int, std::string> > labels = std::vector<std::tuple<int, int, std::string> >(),
                   std::string sampler = "sobol",
                   std::vector<xml_option> sampler_options = {xml_option("integer", "sampleCount", "256"), xml_option("integer", "scramble", "-1")});

    //=========================================================================================================
    //=========================================================================================================
    // Sensor Settings
    chrono::ChVector<> camera_target, camera_origin, camera_up;
    float scale, hour, turbidity, albedo;

    //=========================================================================================================
    // Film settings
    unsigned int height, width;
    //=========================================================================================================
    // Output settings
    std::string data_folder, render_folder;
    //=========================================================================================================
    // Options
    //=========================================================================================================
    xmlTextWriterPtr writer;
};
