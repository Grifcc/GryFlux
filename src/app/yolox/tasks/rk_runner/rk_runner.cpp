#include "rk_runner.h"
#include "package.h"
#include "utils/logger.h"

#define RKNN_CHECK(op, error_msg) \
    do { \
        int ret = (op); \
        if (ret < 0) { \
            LOG.error("%s failed! ret=%d", error_msg, ret); \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

rknn_core_mask NPU_SERIAL_NUM[5] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2, RKNN_NPU_CORE_0_1,
                                    RKNN_NPU_CORE_0_1_2};

namespace GryFlux {
	RkRunner::RkRunner(std::string_view model_path, int npu_id, std::size_t model_width, std::size_t model_height) {
		this->model_width_ = model_width;
		this->model_height_ = model_height;
		LOG.info("model path: %s", model_path.data());
		// load model
		auto model_meta = load_model(model_path);
		if (!model_meta) {
			LOG.error("Fail to read model");
			throw std::runtime_error("fail to read model");
		}
		auto &[model_data, model_data_size] = *model_meta;
		RKNN_CHECK(rknn_init(&rknn_ctx_, model_data.get(), model_data_size, 0, nullptr), "rknn_init");

		// set npu id
		auto core_mask = NPU_SERIAL_NUM[npu_id];
		RKNN_CHECK(rknn_set_core_mask(rknn_ctx_, core_mask), "set NPU core mask");

		// query rknn version
		rknn_sdk_version version;
		RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)), "query rknn version");
		LOG.info("rknn sdk version: %s,driver version: %s", version.api_version, version.drv_version);

		// Get Model Input Output Number
		rknn_input_output_num io_num;
		RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)), "query rknn input/output num");

		input_num_ = io_num.n_input;
		LOG.info("model input num: %d", input_num_);

		// set rknn input/output attr
		input_attrs_ = new rknn_tensor_attr[input_num_];
		memset(input_attrs_, 0, sizeof(rknn_tensor_attr) * input_num_);
		for(std::size_t i = 0; i < input_num_; i++) {
			input_attrs_[i].index = i;
			RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr)), "query rknn input attr");
			this->dump_tensor_attr(&(input_attrs_[i]));
		}

		output_num_ = io_num.n_output;
		LOG.info("output num: %d", output_num_);
		output_attrs_ = new rknn_tensor_attr[output_num_];
		memset(output_attrs_, 0, sizeof(rknn_tensor_attr) * output_num_);
		for(std::size_t i = 0; i < output_num_; i++) {
			output_attrs_[i].index = i;
			RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr)), "query rknn output attr");
			this->dump_tensor_attr(&(output_attrs_[i]));
		}

		// is quant model
		if (output_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs_[0].type != RKNN_TENSOR_FLOAT16) {
			is_quant_ = true;
			LOG.info("model is quantized");
		}
		else {
			throw std::runtime_error("model is not quantized");
			is_quant_ = false;
		}
		// Create input tensor memory
		if (input_attrs_[0].fmt != RKNN_TENSOR_NHWC) {
			throw std::runtime_error("only input type nhwc is supported in zero_copy mode");
		} 
		rknn_tensor_type input_type =
			RKNN_TENSOR_UINT8;  // default input type is int8 (normalize and quantize need compute in outside)
		rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;  // default fmt is NHWC, npu only support NHWC in zero copy mode
		input_attrs_[0].type = input_type;
		input_attrs_[0].fmt = input_layout;
		LOG.info("input_attrs_[0].size_with_stride:%d", input_attrs_[0].size_with_stride);
		input_mems_.emplace_back(rknn_create_mem(rknn_ctx_, input_attrs_[0].size_with_stride));
		RKNN_CHECK(rknn_set_io_mem(rknn_ctx_, input_mems_[0], &input_attrs_[0]), "set input mem");

		// Create output tensor memory
		for (std::size_t i = 0; i < output_num_; ++i) {
			std::size_t output_size;
			// default output type is depend on model, this require float32 to compute top5
			if (is_quant_) {
				output_size = output_attrs_[i].n_elems * sizeof(int8_t);
				output_attrs_[i].type = RKNN_TENSOR_INT8;
			} else {
				output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
				output_size = output_attrs_[i].n_elems * sizeof(float);
			}
			// allocate float32 output tensor
			output_mems_.emplace_back(rknn_create_mem(rknn_ctx_, output_size));
			// set output memory and attribute
			RKNN_CHECK(rknn_set_io_mem(rknn_ctx_, output_mems_[i], &output_attrs_[i]), "set ouput mem");

		}		
	}
    void RkRunner::dump_tensor_attr(rknn_tensor_attr* attr) {
		LOG.info("\tindex=%d, name=%s, \n\t\tn_dims=%d, dims=[%d, %d, %d, %d], \n\t\tn_elems=%d, size=%d, fmt=%s, \n\t\ttype=%s, qnt_type=%s, "
				"zp=%d, scale=%f",
				attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
				attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
				get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
	}


	std::optional<ModelData> RkRunner::load_model(std::string_view filename) {
		ModelData model_meta;
		std::ifstream file(filename.data(), std::ios::binary | std::ios::ate); // Open file in binary mode and seek to the end

		if (!file.is_open()) {
			LOG.error("Failed to open file: %s ", filename.data());
				return std::nullopt;
		}

		const std::size_t fileSize = static_cast<int>(file.tellg()); // Get the file size
		file.seekg(0, std::ios::beg); // Seek back to the beginning
		auto buffer = std::make_unique<unsigned char[]>(fileSize); // Allocate memory for the buffer

		if (!buffer) {
			LOG.error("Memory allocation failed.");
			return std::nullopt;
		}

		file.read(reinterpret_cast<char*>(buffer.get()), fileSize); // Read the entire file into the buffer
		file.close(); // Close the file
		return std::pair{std::move(buffer), fileSize};
	}

    std::shared_ptr<DataObject> RkRunner::process(const std::vector<std::shared_ptr<DataObject>> &inputs) {
		// single input
		if (inputs.size() != 1) return nullptr;

		//set input
        auto input_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
		auto frame = input_data->get_data();
		int height = input_data->get_height();
		int width = input_data->get_width();
		if (width == input_attrs_[0].w_stride) {
		memcpy(input_mems_[0]->virt_addr, frame.data, input_mems_[0]->size);
		}
		else {
			uint8_t *src_ptr = frame.data;
			uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(input_mems_[0]->virt_addr);
			int src_wc_elem = width * 3; // rgb
			int dst_wc_elem = input_attrs_[0].w_stride * 3; // nhwc
			// nhwc
			for (int h = 0; h < height; h++) {
				memcpy(dst_ptr, src_ptr, src_wc_elem);
				src_ptr += src_wc_elem;
				dst_ptr += dst_wc_elem;
			}
		}
		rknn_mem_sync(rknn_ctx_, input_mems_[0], RKNN_MEMORY_SYNC_TO_DEVICE);

		//run inference
		RKNN_CHECK(rknn_run(rknn_ctx_, nullptr), "rknn run inference");

		//sync output
		auto output_data = std::make_shared<RunnerPackage>(model_width_, model_height_);
		for (std::size_t i = 0; i < output_num_; i++) {
			rknn_mem_sync(rknn_ctx_, output_mems_[i], RKNN_MEMORY_SYNC_FROM_DEVICE);
			std::shared_ptr<float[]> output(new float[output_attrs_[i].n_elems]);
			int zp = output_attrs_[i].zp;
			float scale = output_attrs_[i].scale;
			LOG.info("output zp = %d, scale = %f size = %d", zp, scale, output_attrs_[i].n_elems);
			if (is_quant_) {
				for (int j = 0; j < output_attrs_[i].n_elems; j++) {
					output[j] = deqnt_affine_to_f32(
						reinterpret_cast<int8_t*>(output_mems_[i]->virt_addr)[j], zp, scale);
				}
			} else {
				throw std::runtime_error("output type is not quantized, Not Implemented");
			}	
			// add to ouput package
			output_data->push_data({output, output_attrs_[i].n_elems}, {output_attrs_[i].dims[2], output_attrs_[i].dims[3]});

		}
		return output_data;

	}

	inline float RkRunner::deqnt_affine_to_f32(int8_t qnt, int zp, float scale) {
		return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
	}

	RkRunner::~RkRunner() {
		LOG.info(__PRETTY_FUNCTION__);

		for (auto& mem : input_mems_) {
			rknn_destroy_mem(rknn_ctx_, mem);
		}
		for (auto& mem : output_mems_) {
			rknn_destroy_mem(rknn_ctx_, mem);
		}
		delete[] input_attrs_;
		delete[] output_attrs_;
		rknn_destroy(rknn_ctx_);
	}

}