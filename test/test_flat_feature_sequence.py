from pipeline.flat_feature_sequence import generate_train_validate_and_test


def test_generate_train_and_validate():
    train_set, train_target, validate_set, validate_target, test_set, test_target = generate_train_validate_and_test()
    print(len(train_set), len(train_target), len(validate_set), len(validate_target))
    print(train_set[0], train_target[0], validate_set[0], validate_target[0])
