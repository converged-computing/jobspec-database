#include "MitsubaGenerator.h"
#include <chrono/collision/ChCModelBullet.h>
MitsubaGenerator::MitsubaGenerator() {
    writer = 0;

    width = 1920;
    height = 1080;

    camera_target = chrono::ChVector<>(0, -3, 0);
    camera_origin = chrono::ChVector<>(0, -2, 20);
    camera_up = chrono::ChVector<>(0, 1, 0);

    scale = 3;
    hour = 12;
    turbidity = 10;
    albedo = .15;
}
MitsubaGenerator::MitsubaGenerator(const std::string& filename) {
    // create xml header
    MitsubaGenerator();
    writer = xmlNewTextWriterFilename(filename.c_str(), 0);
    xmlTextWriterStartDocument(writer, NULL, "utf-8", NULL);

    xmlTextWriterStartElement(writer, BAD_CAST "scene");
    xmlTextWriterWriteAttribute(writer, BAD_CAST "version", BAD_CAST "0.5.0");
}
void MitsubaGenerator::Open(const std::string& filename) {
    if (writer == 0) {
        writer = xmlNewTextWriterFilename(filename.c_str(), 0);
        xmlTextWriterStartDocument(writer, NULL, "utf-8", NULL);

        xmlTextWriterStartElement(writer, BAD_CAST "scene");
        xmlTextWriterWriteAttribute(writer, BAD_CAST "version", BAD_CAST "0.5.0");
    } else {
        printf("ERROR: Cannot call Open after using constructor with filename\n");
    }
}
void MitsubaGenerator::CreateNewNode(const std::string& type) {
    xmlTextWriterStartElement(writer, BAD_CAST type.c_str());
}
void MitsubaGenerator::CloseNode() {
    xmlTextWriterEndElement(writer);
}

void MitsubaGenerator::AddAttribute(const std::string& name, const std::string& value) {
    xmlTextWriterWriteAttribute(writer, BAD_CAST name.c_str(), BAD_CAST value.c_str());
}
void MitsubaGenerator::AddAttributes(const std::vector<xml_option>& options) {
    for (int i = 0; i < options.size(); i++) {
        CreateNewNode(options[i].type.c_str());
        AddAttribute("name", options[i].parameter.c_str());
        AddAttribute("value", options[i].value.c_str());
        CloseNode();
    }
}
void MitsubaGenerator::CreatePlugin(const std::string& name, const std::string& type, const std::vector<xml_option>& options) {
    CreateNewNode(name.c_str());
    AddAttribute("type", type.c_str());
    AddAttributes(options);
}
void MitsubaGenerator::CreateTransform(const std::string& name) {
    CreateNewNode("transform");
    AddAttribute("name", name.c_str());
}

void MitsubaGenerator::CreateTransform(const chrono::ChVector<>& scale, const chrono::ChVector<>& position, const chrono::ChQuaternion<>& rotation) {
    CreateTransform("toWorld");

    Scale(scale);
    if (rotation != chrono::QUNIT) {
        double angle;
        chrono::ChVector<> axis;
        rotation.Q_to_AngAxis(angle, axis);
        angle = angle * 180.0 / chrono::CH_C_PI;
        Rotate(angle, axis);
    }
    Translate(position);
}

std::string MitsubaGenerator::CreateTriplet(const chrono::ChVector<>& v) {
    std::stringstream ss;
    ss << float(v.x) << ", " << float(v.y) << ", " << float(v.z);
    return ss.str();
}
void MitsubaGenerator::CreateXYZTriplet(const chrono::ChVector<>& v) {
    AddAttribute("x", std::to_string(float(v.x)));
    AddAttribute("y", std::to_string(float(v.y)));
    AddAttribute("z", std::to_string(float(v.z)));
}

void MitsubaGenerator::Translate(const chrono::ChVector<>& vector) {
    CreateNewNode("translate");
    CreateXYZTriplet(vector);

    CloseNode();
}
void MitsubaGenerator::Scale(const chrono::ChVector<>& vector) {
    CreateNewNode("scale");
    CreateXYZTriplet(vector);
    CloseNode();
}
void MitsubaGenerator::Rotate(const double angle, const chrono::ChVector<>& axis) {
    CreateNewNode("rotate");

    CreateXYZTriplet(axis);
    AddAttribute("angle", std::to_string(angle));
    CloseNode();
}

void MitsubaGenerator::LookAt(const chrono::ChVector<>& origin, const chrono::ChVector<>& look_at, const chrono::ChVector<>& up) {
    CreateNewNode("lookAt");

    AddAttribute("origin", CreateTriplet(origin).c_str());
    AddAttribute("target", CreateTriplet(look_at).c_str());
    AddAttribute("up", CreateTriplet(up).c_str());

    CloseNode();
}

void MitsubaGenerator::AddEmitter(std::string emitter,
                                  std::vector<xml_option> emitter_options,
                                  const chrono::ChVector<>& scale,
                                  const chrono::ChVector<>& position,
                                  const chrono::ChQuaternion<>& rotation) {
    CreatePlugin("emitter", emitter, emitter_options);
    CreateTransform(scale, position, rotation);
    CloseNode();
    CloseNode();
}

void MitsubaGenerator::AddIntegrator(std::string integrator, std::vector<xml_option> integrator_options) {
    CreatePlugin("integrator", integrator, integrator_options);
    CloseNode();
}
void MitsubaGenerator::AddInclude(std::string filename) {
    CreateNewNode("include");
    AddAttribute("filename", filename);
    CloseNode();
}
void MitsubaGenerator::CreateScene(bool add_integrator, bool add_sensor, bool add_sky) {
    AddInclude("geometry.xml");

    if (add_integrator) {
        AddIntegrator();  // Default arguments
    }
    if (add_sensor) {
        AddSensor(camera_origin, camera_target, camera_up);
    }
    if (add_sky) {
        // Add the Sky emitter
        std::vector<xml_option> emitter_options = {xml_option("float", "scale", std::to_string(scale)), xml_option("float", "hour", std::to_string(hour)),
                                                   xml_option("float", "turbidity", std::to_string(turbidity)),
                                                   xml_option("spectrum", "albedo", std::to_string(albedo))};
        AddEmitter("sky", emitter_options);
    }
    AddInclude("$frame.xml");
}

void MitsubaGenerator::AddMesh(const std::string& filename,
                               const std::string& material,
                               const chrono::ChVector<>& scale,
                               const chrono::ChVector<>& position,
                               const chrono::ChQuaternion<>& rotation) {
    CreateNewNode("shape");  // Create the root integrator node <integrator ...
    AddAttribute("type", "obj");
    {
        CreateNewNode("string");
        AddAttribute("name", "filename");
        AddAttribute("value", filename);
        CloseNode();
    }
    {
        CreateTransform(scale, position, rotation);
        CloseNode();
    }
    {
        CreateNewNode("ref");
        AddAttribute("id", material);
        CloseNode();
    }
    CloseNode();
}

void MitsubaGenerator::AddShape(const std::string& id,
                                const chrono::ChVector<>& scale,
                                const chrono::ChVector<>& position,
                                const chrono::ChQuaternion<>& rotation) {
    CreateNewNode("shape");  // Create the root integrator node <integrator ...
    AddAttribute("type", "instance");
    CreateNewNode("ref");
    AddAttribute("id", id);
    CloseNode();
    CreateTransform(scale, position, rotation);
    CloseNode();

    CloseNode();
}

void MitsubaGenerator::AddCompleteShape(const std::string& id,
                                        const std::string& material_type,
                                        const chrono::ChVector<>& color,
                                        const chrono::ChVector<>& scale,
                                        const chrono::ChVector<>& position,
                                        const chrono::ChQuaternion<>& rotation) {
    CreateNewNode("shape");
    AddAttribute("type", id);
    CreateTransform(scale, position, rotation);
    CloseNode();

    CreateNewNode("bsdf");
    AddAttribute("type", "diffuse");
    CreateNewNode("srgb");
    AddAttribute("name", "reflectance");
    AddAttribute("value", CreateTriplet(color).c_str());
    CloseNode();
    CloseNode();
    CloseNode();
}

void MitsubaGenerator::AddSimpleShape(const int type,
                                      const chrono::ChVector<>& scale,
                                      const chrono::ChVector<>& position,
                                      const chrono::ChQuaternion<>& rotation) {
    switch (type) {
        case chrono::collision::SPHERE:
            AddShape("sphere", scale, position, rotation);
            break;
        case chrono::collision::ELLIPSOID:
            AddShape("sphere", scale, position, rotation);
            break;
        case chrono::collision::BOX:
            AddShape("cube", scale, position, rotation);
            break;
        case chrono::collision::CYLINDER:
            AddShape("cylinder", scale, position, rotation);
            break;
        case chrono::collision::CONE:
            AddShape("cone", scale, position, rotation);
            break;
        case chrono::collision::CAPSULE:
            AddShape("sphere", chrono::ChVector<>(scale.x), position + rotation.Rotate(chrono::ChVector<>(0, scale.y, 0)), chrono::QUNIT);
            AddShape("cylinder", scale, position, rotation);
            AddShape("sphere", chrono::ChVector<>(scale.x), position + rotation.Rotate(chrono::ChVector<>(0, -scale.y, 0)), chrono::QUNIT);
            break;
        default:
            // type is -1 (triangle mesh)
            break;
    }
}

void MitsubaGenerator::AddSensor(chrono::ChVector<> origin,
                                 chrono::ChVector<> target,
                                 chrono::ChVector<> up,
                                 std::vector<std::tuple<int, int, std::string> > labels,
                                 std::string sampler,
                                 std::vector<xml_option> sampler_options) {
    /////Sensor
    std::vector<xml_option> sensor_options;
    sensor_options.push_back(xml_option("string", "fovAxis", "smaller"));
    sensor_options.push_back(xml_option("float", "fov", "45"));
    CreatePlugin("sensor", "perspective", sensor_options);
    CreateTransform("toWorld");
    LookAt(origin, target, up);
    CloseNode();

    /////Sampler
    CreatePlugin("sampler", sampler, sampler_options);
    CloseNode();

    /////Film
    std::vector<xml_option> film_options;
    film_options.push_back(xml_option("string", "pixelFormat", "rgba"));
    film_options.push_back(xml_option("integer", "width", "1920"));
    film_options.push_back(xml_option("integer", "height", "1080"));
    film_options.push_back(xml_option("boolean", "banner", "false"));

    if (labels.size() > 0) {
        for (int i = 0; i < labels.size(); i++) {
            std::tuple<int, int, std::string> label = labels[i];
            film_options.push_back(
                xml_option("string", "label [" + std::to_string(std::get<0>(label)) + "," + std::to_string(std::get<1>(label)) + "]", std::get<2>(label)));
        }
    }
    CreatePlugin("film", "ldrfilm", film_options);
    CloseNode();
    CloseNode();
}

void MitsubaGenerator::Write() {
    xmlTextWriterEndDocument(writer);
    xmlFreeTextWriter(writer);
}

void MitsubaGenerator::ExportDriver(const std::string& filename) {}
