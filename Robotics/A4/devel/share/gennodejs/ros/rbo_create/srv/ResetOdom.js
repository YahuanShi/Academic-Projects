// Auto-generated. Do not edit!

// (in-package rbo_create.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class ResetOdomRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.reset_odom = null;
    }
    else {
      if (initObj.hasOwnProperty('reset_odom')) {
        this.reset_odom = initObj.reset_odom
      }
      else {
        this.reset_odom = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ResetOdomRequest
    // Serialize message field [reset_odom]
    bufferOffset = _serializer.bool(obj.reset_odom, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ResetOdomRequest
    let len;
    let data = new ResetOdomRequest(null);
    // Deserialize message field [reset_odom]
    data.reset_odom = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'rbo_create/ResetOdomRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '75845a070de97808c5df63dd2cc872a3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool reset_odom
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ResetOdomRequest(null);
    if (msg.reset_odom !== undefined) {
      resolved.reset_odom = msg.reset_odom;
    }
    else {
      resolved.reset_odom = false
    }

    return resolved;
    }
};

class ResetOdomResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ResetOdomResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ResetOdomResponse
    let len;
    let data = new ResetOdomResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'rbo_create/ResetOdomResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ResetOdomResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: ResetOdomRequest,
  Response: ResetOdomResponse,
  md5sum() { return 'd097c2ef742c9464ee3a0e656d7572a2'; },
  datatype() { return 'rbo_create/ResetOdom'; }
};
