    import sys
    import tensorflow as tf
    from tensorflow.core.util import event_pb2
    import os

    # Suppress TensorFlow INFO and WARNING messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def read_event_file(event_file_path):
        """
        Reads a TensorBoard event file and prints its scalar summaries.
        """
        if not os.path.exists(event_file_path):
            print(f"Error: File not found at '{event_file_path}'")
            return

        print(f"--- Reading data from: {event_file_path} ---\n")
        try:
            # Use tf.compat.v1.train.summary_iterator to safely iterate through the events
            serialized_examples = tf.compat.v1.train.summary_iterator(event_file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            print("Please ensure the file path is correct and it is a valid TensorFlow event file.")
            return

        count = 0
        for serialized_example in serialized_examples:
            try:
                event = event_pb2.Event.FromString(serialized_example)
                for value in event.summary.value:
                    if value.HasField('simple_value'):  # This checks if it's a scalar value
                        tag = value.tag
                        step = event.step
                        scalar_value = value.simple_value
                        print(f"Step: {step:<8} | Tag: {tag:<30} | Value: {scalar_value:.6f}")
                        count += 1
            except Exception as e:
                # This can happen if an event record is corrupted
                print(f"Warning: Could not parse an event record. Error: {e}")
        
        if count == 0:
            print("No valid scalar summaries were found in the file.")
        else:
            print(f"\n--- Found {count} scalar data points. ---")


    if __name__ == "__main__":
        if len(sys.argv) != 2:
            print("Usage: python read_events.py <path_to_event_file>")
        else:
            file_path = sys.argv[1]
            read_event_file(file_path)
